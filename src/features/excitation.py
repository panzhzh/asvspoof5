"""
Excitation feature extraction (E, 32-d) and voiced mask (V) aligned to 20 ms frames.

Design goals (stage 0/1):
- Single STFT (n_fft=512, win=400@16k, hop=320) reused by all sub-features
- Return E [T, 32] (float32 by default; caller may cast to float16 for storage)
- Return V [T] (uint8 0/1) from voicing probability thresholding

Notes:
- This is a pragmatic GPU-first implementation; some features (e.g., vibrato) are
  approximated conservatively to keep the dependency surface small.
- F0 is estimated via a cepstrum-like method from log-magnitude spectra.
- Group-delay is approximated via phase-difference along frequency bins.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class EConfig:
    sr: int = 16000
    n_fft: int = 512
    win_length: int = 400      # 25 ms @ 16 kHz
    hop_length: int = 320      # 20 ms @ 16 kHz
    center: bool = True
    fmin_hz: float = 80.0
    fmax_hz: float = 500.0
    voiced_th: float = 0.5


def _stft(wav: torch.Tensor, cfg: EConfig) -> torch.Tensor:
    """Compute complex STFT (frames x freq). wav: (T,) float32 on CPU/GPU."""
    device = wav.device
    window = torch.hann_window(cfg.win_length, device=device, dtype=wav.dtype)
    stft = torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=cfg.center,
        return_complex=True,
    )  # (freq, frames)
    return stft.transpose(0, 1).contiguous()  # (frames, freq)


def _freq_grid(cfg: EConfig, device, dtype) -> torch.Tensor:
    return torch.linspace(0, cfg.sr / 2, cfg.n_fft // 2 + 1, device=device, dtype=dtype)


def _band_mask(freq: torch.Tensor, f_lo: float, f_hi: float) -> torch.Tensor:
    return (freq >= f_lo) & (freq < f_hi)


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp_min(x, eps))


def _percentiles(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute percentiles along last dim using torch.kthvalue approximation."""
    # x: (..., N) ; q in [0,1]
    x_sorted, _ = torch.sort(x, dim=-1)
    idx = torch.clamp((q * (x.shape[-1] - 1)).round().long(), min=0, max=x.shape[-1] - 1)
    # Gather last-dim values for each q
    out = []
    for qi in idx:
        out.append(x_sorted[..., qi])
    return torch.stack(out, dim=-1)


def _cepstrum_peak_f0(log_mag: torch.Tensor, cfg: EConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate F0 (log-scale) and voicing probability from log-magnitude spectra.
    log_mag: (T, F)
    returns: logF0 (T,), voicing_prob (T,)
    """
    T, F = log_mag.shape
    device = log_mag.device
    # real cepstrum via irfft over frequency (mirror spectrum)
    # build symmetric spectrum for irfft: input is positive freqs; use irfft
    # Note: irfft expects Hermitian symmetry, but we use log|X| which is real.
    ceps = torch.fft.irfft(log_mag, n=(2 * (F - 1)), dim=-1)  # (T, n_time)
    # Quefrency axis in seconds (k / sr)
    n_time = ceps.shape[-1]
    q_idx_min = max(1, int(cfg.sr / cfg.fmax_hz))   # ~ 32 for 500 Hz
    q_idx_max = min(n_time - 1, int(cfg.sr / cfg.fmin_hz))  # ~ 200 for 80 Hz
    if q_idx_max <= q_idx_min + 1:
        # Fallback: no reliable F0 band
        logf0 = torch.zeros(T, device=device)
        vprob = torch.zeros(T, device=device)
        return logf0, vprob
    band = ceps[:, q_idx_min:q_idx_max]
    peak_vals, peak_pos = torch.max(band, dim=-1)
    # Map quefrency index to Hz and to logF0
    q_pos = (q_idx_min + peak_pos).float()  # in samples
    f0 = torch.clamp(cfg.sr / q_pos, min=1.0)
    logf0 = _safe_log(f0)
    # Simple voicing proxy: normalized peak within band (0..1)
    band_min, _ = torch.min(band, dim=-1)
    band_max, _ = torch.max(band, dim=-1)
    denom = torch.clamp_min(band_max - band_min, 1e-6)
    vprob = torch.clamp((peak_vals - band_min) / denom, 0.0, 1.0)
    return logf0, vprob


def _linear_regression(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return slope of simple linear regression y ~ a * x + b along last dim.
    x,y: (..., N)
    """
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    xm = x - x_mean
    ym = y - y_mean
    num = (xm * ym).sum(dim=-1)
    den = torch.clamp_min((xm * xm).sum(dim=-1), 1e-8)
    return num / den


def _spectral_flatness(mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Geometric mean / arithmetic mean along last dim."""
    gm = torch.exp(_safe_log(mag, eps).mean(dim=-1))
    am = torch.clamp_min(mag.mean(dim=-1), eps)
    return torch.clamp(gm / am, min=0.0)


def _group_delay(phase: torch.Tensor, cfg: EConfig) -> torch.Tensor:
    """Approximate group delay via wrapped phase difference along frequency axis.
    phase: (T,F) in radians. Returns gd (T,F-1).
    We avoid torch.unwrap for compatibility: compute diff then wrap to (-pi, pi].
    """
    dphi = torch.diff(phase, dim=-1)
    # wrap to (-pi, pi]
    dphi_wrapped = (dphi + math.pi) % (2 * math.pi) - math.pi
    return -dphi_wrapped


def extract_excitation_gpu(wav: torch.Tensor, cfg: EConfig | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract E[frames,32] (float32) and V[frames] (uint8) aligned to STFT frames.

    wav: (T,) float32 tensor on CPU/GPU, 16kHz mono.
    returns: (E, V)
    """
    if cfg is None:
        cfg = EConfig()
    device = wav.device
    stft = _stft(wav, cfg)  # (T, F)
    T, Fpos = stft.shape
    # magnitude/phase for positive freqs
    pos = stft[:, : cfg.n_fft // 2 + 1]
    mag = pos.abs()
    log_mag = _safe_log(mag)
    phase = torch.angle(pos)
    freq = _freq_grid(cfg, device, wav.dtype)

    # 1) F0 & voicing
    logf0, vprob = _cepstrum_peak_f0(log_mag, cfg)
    vmask = (vprob >= cfg.voiced_th).to(torch.uint8)
    # jitter (relative): |Δ logf0| / (|logf0|+eps)
    d_logf0 = torch.zeros_like(logf0)
    d_logf0[1:] = (logf0[1:] - logf0[:-1]).abs()
    jitter_rel = d_logf0 / (logf0.abs() + 1e-4)

    # 2) HNR proxy from cepstrum peak contrast
    # Already have peak_vals via _cepstrum_peak_f0; recompute quickly
    ceps = torch.fft.irfft(log_mag, n=(2 * (Fpos - 1)), dim=-1)
    q_idx_min = max(1, int(cfg.sr / cfg.fmax_hz))
    q_idx_max = min(ceps.shape[-1] - 1, int(cfg.sr / cfg.fmin_hz))
    band = ceps[:, q_idx_min:q_idx_max]
    peak_vals, _ = torch.max(band, dim=-1)
    hnr_proxy = peak_vals - band.mean(dim=-1)

    # 3) spectral slope (0.3–3 kHz)
    mask_03_3k = _band_mask(freq, 300.0, 3000.0)
    xf = _safe_log(freq[mask_03_3k] + 1e-3)  # avoid log(0)
    yf = log_mag[:, mask_03_3k]
    slope = _linear_regression(xf.expand(T, -1), yf)

    # 4) spectral flatness (0–1k, 1–4k)
    mask_0_1k = _band_mask(freq, 0.0, 1000.0)
    mask_1_4k = _band_mask(freq, 1000.0, 4000.0)
    flat_0_1k = _spectral_flatness(mag[:, mask_0_1k])
    flat_1_4k = _spectral_flatness(mag[:, mask_1_4k])

    # 5) group delay stats over 1–4 kHz
    gd = _group_delay(phase, cfg)  # (T, Fpos-1)
    freq_mid = freq[:-1]
    mask_gd = _band_mask(freq_mid, 1000.0, 4000.0)
    gd_band = gd[:, mask_gd]
    gd_mean = gd_band.mean(dim=-1)
    gd_var = gd_band.var(dim=-1)
    gd_p25, gd_p75 = _percentiles(gd_band, torch.tensor([0.25, 0.75], device=device, dtype=gd_band.dtype)).unbind(-1)

    # 6) energy and band energies/flatness
    energy_log = _safe_log((mag ** 2).sum(dim=-1) + 1e-8)
    # 8 bands up to Nyquist
    bands = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000)]
    band_logE = []
    band_flat = []
    for lo, hi in bands:
        m = _band_mask(freq, float(lo), float(hi))
        if m.sum() == 0:
            be = torch.zeros(T, device=device, dtype=mag.dtype)
            bf = torch.zeros(T, device=device, dtype=mag.dtype)
        else:
            mag_b = mag[:, m]
            be = _safe_log((mag_b ** 2).sum(dim=-1) + 1e-8)
            bf = _spectral_flatness(mag_b)
        band_logE.append(be)
        band_flat.append(bf)
    band_logE = torch.stack(band_logE, dim=-1)  # (T,8)
    band_flat = torch.stack(band_flat, dim=-1)   # (T,8)

    # 7) centroid & rolloff(0.85)
    fcol = freq.view(1, -1).expand(T, -1)
    centroid = (mag * fcol).sum(dim=-1) / torch.clamp_min(mag.sum(dim=-1), 1e-8)
    # rolloff: smallest freq where cumulative energy >= 85%
    cumE = (mag ** 2).cumsum(dim=-1)
    thr = 0.85 * cumE[..., -1:]
    roll_idx = torch.argmax((cumE >= thr).to(torch.int32), dim=-1)
    rolloff = freq[roll_idx]

    # Assemble E (32 dims)
    feats = [
        logf0, vprob, jitter_rel,
        torch.zeros_like(logf0),  # vibrato_extent (placeholder)
        torch.zeros_like(logf0),  # vibrato_rate (placeholder)
        hnr_proxy, slope, flat_0_1k, flat_1_4k, energy_log,
        gd_mean, gd_var, gd_p25, gd_p75,
        band_logE, band_flat,
        centroid, rolloff,
    ]
    # Stack with proper broadcasting
    E = torch.cat([
        feats[0].unsqueeze(-1), feats[1].unsqueeze(-1), feats[2].unsqueeze(-1),
        feats[3].unsqueeze(-1), feats[4].unsqueeze(-1),
        feats[5].unsqueeze(-1), feats[6].unsqueeze(-1), feats[7].unsqueeze(-1), feats[8].unsqueeze(-1), feats[9].unsqueeze(-1),
        feats[10].unsqueeze(-1), feats[11].unsqueeze(-1), feats[12].unsqueeze(-1), feats[13].unsqueeze(-1),
        feats[14], feats[15],
        feats[16].unsqueeze(-1), feats[17].unsqueeze(-1),
    ], dim=-1)
    # Sanity: if dims mismatch, pad or trim to 32
    if E.shape[-1] < 32:
        pad = torch.zeros(T, 32 - E.shape[-1], device=device, dtype=E.dtype)
        E = torch.cat([E, pad], dim=-1)
    elif E.shape[-1] > 32:
        E = E[:, :32]

    return E.to(torch.float32), vmask
