#!/usr/bin/env python3
"""
One-click C²S pipeline runner (all-in-one; no other scripts required).

Stages (auto-skip if outputs exist unless --force):
  1) Ensure E/V caches for required splits (train + scoring splits)
  2) Fit PCA(64) + KMeans(K) on train-bonafide
  3) Fit per-bucket linear–Gaussian (E|C)
  4) Score splits (voiced-only NLL → window → utter)
  5) Calibrate on chosen split and print metrics (evaluation-package)

Usage examples:
  python scripts/c2s_run.py --config config/config.py --splits eval --calib-split eval
  python scripts/c2s_run.py --k 200 --fit-sample-ratio 0.2 --force-buckets
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def exist_nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


# ------------------------------
# Utilities used by all stages
# ------------------------------

def _json_loads_relaxed(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        fixed = line.replace("\"float16'\>", '"float16"').replace("\"uint8'\>", '"uint8"')
        return json.loads(fixed)


class _RaggedIndex:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.idx = self.root / "index.jsonl"
        self.records: dict[str, dict] = {}
        self.mm: dict[str, np.memmap] = {}
        with self.idx.open() as f:
            for line in f:
                r = _json_loads_relaxed(line)
                self.records[r["utt_id"]] = r

    def _get_mm(self, shard: str) -> np.memmap:
        m = self.mm.get(shard)
        if m is None:
            m = np.lib.format.open_memmap(self.root / shard, mode='r')
            self.mm[shard] = m
        return m

    def get(self, uid: str) -> np.ndarray:
        r = self.records[uid]
        mm = self._get_mm(r["shard"])
        off = int(r["offset_elems"])
        cnt = int(r["elem_count"])
        T = int(r.get("T", r.get("real_len")))
        D = int(r.get("D", max(1, cnt // max(1, T))))
        arr = np.asarray(mm[off:off+cnt]).reshape(T, D)
        return arr


def _make_segments(n_frames: int, win_frames: int = 200, hop_frames: int = 100):
    segs = []
    s = 0
    while s + 1 < n_frames:
        e = min(n_frames, s + win_frames)
        segs.append((s, e))
        if e >= n_frames:
            break
        s = s + hop_frames
    return segs


def _trimmed_mean_t(x, trim: float = 0.1):
    import torch
    if x.numel() == 0:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)
    n = x.numel()
    k = int(n * trim)
    if k <= 0:
        return x.mean()
    xs, _ = torch.sort(x)
    return xs[k: n - k].mean() if (n - 2 * k) > 0 else xs.mean()


# ------------------------------
# Stage 1: E/V extraction
# ------------------------------

def _need_extract_ev(feat_root: Path, splits: Iterable[str]) -> bool:
    for sp in splits:
        if not (feat_root / "E" / sp / "index.jsonl").exists():
            return True
        if not (feat_root / "V" / sp / "index.jsonl").exists():
            return True
    return False


def _extract_e_v(cfg: dict, splits: Iterable[str]) -> None:
    print(f"[1/5] Extracting E/V for splits: {' '.join(splits)}")
    # Lazy imports
    import soundfile as sf
    import librosa
    import torch
    from src.features.excitation import extract_excitation_gpu, EConfig
    from src.data.e_writer import RaggedMemmapWriter, ShardSpec

    feat_root = Path(cfg["feature_path"])  # data/ASVspoof5/features
    db_root = Path(cfg["database_path"])  # data/ASVspoof5

    def tsv_audio_for_split(sp: str) -> tuple[Path, Path]:
        if sp == 'train':
            return db_root / "ASVspoof5.train.tsv", db_root / "flac_T"
        if sp == 'dev':
            return db_root / "ASVspoof5.dev.track_1.tsv", db_root / "flac_D"
        if sp == 'eval':
            return db_root / "ASVspoof5.eval.track_1.tsv", db_root / "flac_E"
        raise ValueError(f"Unknown split {sp}")

    def read_ids(tsv_path: Path) -> list[str]:
        ids = []
        with tsv_path.open() as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 2:
                    ids.append(p[1])
        return ids

    def load_wav(path: Path) -> np.ndarray:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = librosa.to_mono(wav.T)
        if sr != 16000:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        return np.clip(wav.astype(np.float32), -1.0, 1.0)

    e_cfg = EConfig(sr=16000, n_fft=512,
                    win_length=400, hop_length=320, center=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for sp in splits:
        tsv_path, wav_dir = tsv_audio_for_split(sp)
        if not wav_dir.exists():
            print(f"[WARN] audio dir missing for {sp}: {wav_dir}")
            continue
        ids = read_ids(tsv_path)
        print(f"Split {sp}: {len(ids)} files")
        e_spec = ShardSpec(root=feat_root / "E" / sp, dtype=np.float16, target_gb=2.0)
        v_spec = ShardSpec(root=feat_root / "V" / sp, dtype=np.uint8, target_gb=2.0)
        ew = RaggedMemmapWriter(e_spec, {"L": 1, "sr": 16000, "win_ms": 25, "stride_ms": 20, "layout": "TD"})
        vw = RaggedMemmapWriter(v_spec, {"L": 1, "sr": 16000, "win_ms": 25, "stride_ms": 20, "layout": "TD"})

        mean = None
        M2 = None
        count = 0
        for uid in ids:
            p = wav_dir / f"{uid}.flac"
            if not p.exists():
                p = wav_dir / f"{uid}.wav"
                if not p.exists():
                    print(f"[MISS] {p}")
                    continue
            wav = load_wav(p)
            wt = torch.from_numpy(wav).to(device)
            with torch.no_grad():
                E, V = extract_excitation_gpu(wt, e_cfg)
            E = E.cpu().numpy().astype(np.float32)
            V = V.cpu().numpy().astype(np.uint8)
            T = int(E.shape[0])
            De = int(E.shape[1])
            if mean is None:
                mean = E.sum(axis=0)
                M2 = (E ** 2).sum(axis=0)
                count = T
            else:
                mean += E.sum(axis=0)
                M2 += (E ** 2).sum(axis=0)
                count += T
            ew.write_sample(uid, E.astype(np.float16).ravel(), T=T, D=De)
            vw.write_sample(uid, V.ravel(), T=T, D=1)
        ew.close(); vw.close()
        if count > 0:
            mu = (mean / count).astype(np.float32)
            var = (M2 / count) - mu ** 2
            var = np.maximum(var, 1e-8)
            std = np.sqrt(var).astype(np.float32)
            stats = {"mean": mu.tolist(), "std": std.tolist(), "count_frames": int(count)}
            (e_spec.root / "stats.json").write_text(json.dumps(stats))
            print(f"Saved stats for {sp}: frames={count}")


# ------------------------------
# Stage 2: Buckets (PCA+KMeans)
# ------------------------------

def _fit_buckets(cfg: dict, out_dir: Path, k: int, sample_ratio: float) -> None:
    from src.data.datasets import genSpoof_list, FeatureLoader
    from src.models.c2s_bucket import _fit_incremental_pca, _save_pca_npz, _fit_minibatch_kmeans, _save_kmeans_npz

    db_root = Path(cfg["database_path"])  # data/ASVspoof5
    feat_root = Path(cfg["feature_path"]) / "train"
    d_label, file_train = genSpoof_list(dir_meta=db_root / "ASVspoof5.train.tsv", is_train=True, is_eval=False)
    bon_ids = [u for u in file_train if d_label.get(u, 0) == 1]
    loader = FeatureLoader(feat_root)
    rng = np.random.default_rng(0)

    def iter_frames():
        buf = []
        acc = 0
        for uid in bon_ids:
            C_ltd = loader.get_features(uid)
            C_td = C_ltd.mean(axis=0).astype(np.float32)
            T = C_td.shape[0]
            if T <= 0:
                continue
            n = max(1, int(T * sample_ratio))
            idx = rng.choice(T, size=n, replace=False)
            X = C_td[idx]
            buf.append(X)
            acc += X.shape[0]
            if acc >= 200000:
                Y = np.concatenate(buf, axis=0)
                yield Y
                buf = []; acc = 0
        if buf:
            Y = np.concatenate(buf, axis=0)
            yield Y

    mean, comps = _fit_incremental_pca(iter_frames(), out_dim=64)
    _save_pca_npz(out_dir / "pca64.npz", mean, comps)
    def proj_iter():
        for X in iter_frames():
            X = np.asarray(X, dtype=np.float32)
            yield (X - mean) @ comps.T
    cents = _fit_minibatch_kmeans(proj_iter(), k=int(k))
    _save_kmeans_npz(out_dir / f"kmeans_k{k}.npz", cents)


# ------------------------------
# Stage 3: Linear–Gaussian fitting
# ------------------------------

def _fit_lin_gauss(cfg: dict, pca_npz: Path, kmeans_npz: Path, out_npz: Path, alpha: float, var_floor: float) -> None:
    from src.data.datasets import genSpoof_list, FeatureLoader
    import torch

    def load_npz(path: Path) -> dict:
        d = np.load(path)
        return {k: d[k] for k in d.files}

    db_root = Path(cfg["database_path"])  # data/ASVspoof5
    feat_root = Path(cfg["feature_path"]) / "train"
    e_root = Path(cfg["feature_path"]) / "E" / "train"
    v_root = Path(cfg["feature_path"]) / "V" / "train"

    d_label, file_train = genSpoof_list(dir_meta=db_root / "ASVspoof5.train.tsv", is_train=True, is_eval=False)
    bon_ids = [u for u in file_train if d_label.get(u, 0) == 1]

    pca = load_npz(pca_npz)
    kmeans = load_npz(kmeans_npz)
    mean = torch.tensor(pca["mean"], dtype=torch.float32)
    comps = torch.tensor(pca["components"], dtype=torch.float32)
    cents = torch.tensor(kmeans["centroids"], dtype=torch.float32)

    K = int(cents.shape[0]); D64 = int(comps.shape[0]); De = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = mean.to(device); comps = comps.to(device); cents = cents.to(device)

    dX = D64 + 1
    XT_X = torch.zeros((K, dX, dX), dtype=torch.float64, device=device)
    XT_Y = torch.zeros((K, dX, De), dtype=torch.float64, device=device)
    counts = torch.zeros((K,), dtype=torch.int64, device=device)

    c_loader = FeatureLoader(feat_root)
    e_index = _RaggedIndex(e_root)
    v_index = _RaggedIndex(v_root)

    # Pass 1: accumulate
    for uid in bon_ids:
        try:
            C_ltd = c_loader.get_features(uid)
            C_td = C_ltd.mean(axis=0).astype(np.float32)
            E_td = e_index.get(uid).astype(np.float32)
            V_t = v_index.get(uid).astype(np.uint8).reshape(-1)
        except Exception:
            continue
        T = min(C_td.shape[0], E_td.shape[0], V_t.shape[0])
        if T <= 0:
            continue
        C = torch.from_numpy(C_td[:T]).to(device)
        E = torch.from_numpy(E_td[:T]).to(device)
        V = torch.from_numpy(V_t[:T].astype(np.int64)).to(device)
        mask = V > 0
        if mask.sum() == 0:
            continue
        C = C[mask]; E = E[mask]
        C64 = (C - mean) @ comps.T
        x2 = (C64 * C64).sum(dim=1, keepdim=True)
        c2 = (cents * cents).sum(dim=1).view(1, -1)
        d2 = x2 + c2 - 2.0 * (C64 @ cents.T)
        bidx = torch.argmin(d2, dim=1)
        ones = torch.ones((C64.shape[0], 1), dtype=C64.dtype, device=device)
        X = torch.cat([C64, ones], dim=1).to(torch.float64)
        Y = E.to(torch.float64)
        for k in bidx.unique().tolist():
            sel = (bidx == k)
            Xk = X[sel]; Yk = Y[sel]
            XT_X[k] += Xk.T @ Xk
            XT_Y[k] += Xk.T @ Yk
            counts[k] += int(sel.sum().item())

    A = torch.zeros((K, De, D64), dtype=torch.float32, device=device)
    b = torch.zeros((K, De), dtype=torch.float32, device=device)
    for k in range(K):
        if counts[k] <= 0:
            continue
        XX = XT_X[k].clone()
        XX += alpha * torch.eye(dX, dtype=XX.dtype, device=device)
        XY = XT_Y[k]
        try:
            W = torch.linalg.solve(XX, XY)
        except RuntimeError:
            W, _ = torch.lstsq(XY, XX)
        W = W.to(torch.float32)
        A[k] = W[:-1, :].T
        b[k] = W[-1, :].T

    var_acc = torch.zeros((K, De), dtype=torch.float64, device=device)
    var_cnt = torch.zeros((K,), dtype=torch.int64, device=device)
    for uid in bon_ids:
        try:
            C_ltd = c_loader.get_features(uid)
            C_td = C_ltd.mean(axis=0).astype(np.float32)
            E_td = e_index.get(uid).astype(np.float32)
            V_t = v_index.get(uid).astype(np.uint8).reshape(-1)
        except Exception:
            continue
        T = min(C_td.shape[0], E_td.shape[0], V_t.shape[0])
        if T <= 0:
            continue
        C = torch.from_numpy(C_td[:T]).to(device)
        E = torch.from_numpy(E_td[:T]).to(device)
        V = torch.from_numpy(V_t[:T].astype(np.int64)).to(device)
        mask = V > 0
        if mask.sum() == 0:
            continue
        C = C[mask]; E = E[mask]
        C64 = (C - mean) @ comps.T
        x2 = (C64 * C64).sum(dim=1, keepdim=True)
        c2 = (cents * cents).sum(dim=1).view(1, -1)
        d2 = x2 + c2 - 2.0 * (C64 @ cents.T)
        bidx = torch.argmin(d2, dim=1)
        for k in bidx.unique().tolist():
            sel = (bidx == k)
            if counts[k] <= 0 or sel.sum() == 0:
                continue
            Ek = E[sel]; C64k = C64[sel]
            pred = (C64k @ A[k].T) + b[k]
            resid = Ek - pred
            var_acc[k] += (resid.to(torch.float64) ** 2).mean(dim=0)
            var_cnt[k] += 1

    sigma = torch.sqrt(torch.clamp(var_acc / torch.clamp(var_cnt.view(-1, 1).to(torch.float64), min=1.0), min=var_floor)).to(torch.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz,
             A=A.detach().cpu().numpy(),
             b=b.detach().cpu().numpy(),
             sigma=sigma.detach().cpu().numpy(),
             centroids=cents.detach().cpu().numpy(),
             pca_mean=mean.detach().cpu().numpy(),
             pca_components=comps.detach().cpu().numpy())
    print(f"Saved linGauss model to {out_npz}")


# ------------------------------
# Stage 4: Scoring
# ------------------------------

def _score_splits(cfg: dict, lin_gauss_npz: Path, splits: List[str], out_path: Path, trim: float, skip_unvoiced_ratio: float, batch_utts: int = 16) -> None:
    from src.data.datasets import genSpoof_list, FeatureLoader
    import torch

    d = np.load(lin_gauss_npz)
    A = torch.tensor(d["A"], dtype=torch.float32)
    b = torch.tensor(d["b"], dtype=torch.float32)
    sigma = torch.tensor(d["sigma"], dtype=torch.float32)
    cents = torch.tensor(d["centroids"], dtype=torch.float32)
    mean = torch.tensor(d["pca_mean"], dtype=torch.float32)
    comps = torch.tensor(d["pca_components"], dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A=A.to(device); b=b.to(device); sigma=sigma.to(device); cents=cents.to(device); mean=mean.to(device); comps=comps.to(device)

    feat_root = Path(cfg["feature_path"])  # features
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open('w')
    for split in splits:
        db_root = Path(cfg["database_path"])  # data
        if split == 'dev':
            d_label, ids = genSpoof_list(dir_meta=db_root / "ASVspoof5.dev.track_1.tsv", is_train=False, is_eval=False)
            ids = list(d_label.keys())
        elif split == 'eval':
            ids = genSpoof_list(dir_meta=db_root / "ASVspoof5.eval.track_1.tsv", is_train=False, is_eval=True)
        else:
            raise ValueError(f"Unknown split {split}")
        c_loader = FeatureLoader(feat_root / split)
        e_index = _RaggedIndex(feat_root / "E" / split)
        v_index = _RaggedIndex(feat_root / "V" / split)
        # Batched over utterances to better utilize GPU
        for i in range(0, len(ids), max(1, int(batch_utts))):
            batch_ids = ids[i:i+max(1, int(batch_utts))]
            C_list = []
            E_list = []
            V_list = []
            meta = []  # (uid, T)
            for uid in batch_ids:
                try:
                    C_ltd = c_loader.get_features(uid)
                    C_td = C_ltd.mean(axis=0).astype(np.float32)
                    E_td = e_index.get(uid).astype(np.float32)
                    V_t = v_index.get(uid).astype(np.uint8).reshape(-1)
                except Exception:
                    continue
                T = min(C_td.shape[0], E_td.shape[0], V_t.shape[0])
                if T <= 0:
                    continue
                C_list.append(C_td[:T])
                E_list.append(E_td[:T])
                V_list.append(V_t[:T])
                meta.append((uid, T))
            if not meta:
                continue
            # Concatenate over frames
            C_all = torch.from_numpy(np.concatenate(C_list, axis=0)).to(device)
            E_all = torch.from_numpy(np.concatenate(E_list, axis=0)).to(device)
            V_all = torch.from_numpy(np.concatenate([v.astype(np.int64) for v in V_list], axis=0)).to(device)
            # Project and assign buckets for all frames at once
            C64 = (C_all - mean) @ comps.T
            x2 = (C64 * C64).sum(dim=1, keepdim=True)
            c2 = (cents * cents).sum(dim=1).view(1, -1)
            d2 = x2 + c2 - 2.0 * (C64 @ cents.T)
            bidx = torch.argmin(d2, dim=1)
            # Compute NLL per frame (vectorized by bucket)
            nll_all = torch.empty((C64.shape[0],), dtype=torch.float32, device=device)
            for k in bidx.unique().tolist():
                sel = (bidx == k)
                if sel.sum() == 0:
                    continue
                C64k = C64[sel]
                Ek = E_all[sel]
                pred = (C64k @ A[k].T) + b[k]
                sig = sigma[k]
                z = (Ek - pred) / (sig + 1e-6)
                quad = 0.5 * (z * z).sum(dim=1)
                logdet = 0.5 * torch.log((sig * sig) + 1e-12).sum()
                nll_all[sel] = quad + logdet
            # Slice back per-utterance and aggregate to a score
            offset = 0
            for (uid, T), V_np in zip(meta, V_list):
                s = offset; e = offset + T
                offset = e
                V = torch.from_numpy(V_np.astype(np.int64)).to(device)
                nll_t = nll_all[s:e]
                segs = _make_segments(T, win_frames=200, hop_frames=100)
                wins = []
                for ss, ee in segs:
                    if ee <= ss:
                        continue
                    v = V[ss:ee]
                    if (v == 0).sum().item() / max(1, (ee - ss)) >= skip_unvoiced_ratio:
                        continue
                    wins.append(_trimmed_mean_t(nll_t[ss:ee], trim))
                score = float(nll_t.mean().item()) if len(wins) == 0 else float(torch.stack(wins).mean().item())
                fout.write(json.dumps({"utt_id": uid, "split": split, "score": score, "n_windows": len(wins)}) + "\n")
    fout.close()
    print(f"Saved C²S raw scores to {out_path}")


# ------------------------------
# Stage 5: Calibration + metrics
# ------------------------------

def _calibrate_and_eval(cfg: dict, scores_path: Path, out_jsonl: Path, split: str) -> None:
    db_root = Path(cfg["database_path"])  # data/ASVspoof5
    if split == 'dev':
        tsv = db_root / "ASVspoof5.dev.track_1.tsv"
    else:
        tsv = db_root / "ASVspoof5.eval.track_1.tsv"

    # Load scores for the split
    m = {}
    with scores_path.open() as f:
        for line in f:
            r = json.loads(line)
            if str(r.get("split", split)) != split:
                continue
            m[str(r["utt_id"]) ] = float(r["score"])
    # Load labels
    lab = {}
    with tsv.open() as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 9:
                lab[p[1]] = 1 if p[8] == 'bonafide' else 0
    ids = [u for u in m.keys() if u in lab]
    x = np.array([m[u] for u in ids], dtype=np.float64)
    y = np.array([lab[u] for u in ids], dtype=np.float64)
    print(f"Dev samples for calibration: {len(ids)}")

    # Fit logistic regression (simple GD)
    a = 0.0; b = 0.0; lr = 0.05; l2 = 1e-3
    for _ in range(2000):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        g_a = np.mean((p - y) * x) + l2 * a
        g_b = np.mean(p - y)
        a -= lr * g_a
        b -= lr * g_b
    print(f"Calib parameters: a={a:.4f}, b={b:.4f}")

    # Write calibrated jsonl
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open('w') as f:
        for u in ids:
            s = float(a * m[u] + b)
            f.write(json.dumps({"utt_id": u, "split": split, "score": s}) + "\n")
    print(f"Saved calibrated {split} scores to {out_jsonl}")
    # Two-col file + keys
    two_col = out_jsonl.with_suffix('.txt')
    with two_col.open('w') as f:
        f.write("filename\tcm-score\n")
        for u in ids:
            f.write(f"{u}\t{a * m[u] + b}\n")
    key_path = two_col.parent / f"{split}_keys.tsv"
    with tsv.open() as fi, key_path.open('w') as fo:
        fo.write("filename\tcm-label\n")
        file_set = set(ids)
        for line in fi:
            p = line.strip().split()
            if len(p) >= 9 and p[1] in file_set:
                fo.write(f"{p[1]}\t{p[8]}\n")

    # print metrics via evaluation-package
    eval_dir = Path(__file__).resolve().parents[1] / "evaluation-package"
    print("\nEvaluation metrics (from evaluation-package):\n")
    subprocess.check_call([sys.executable, "evaluation.py", "--m", "t1", "--cm", str(two_col), "--cm_keys", str(key_path)], cwd=eval_dir)


def main():
    ap = argparse.ArgumentParser(description="Run full C²S pipeline end-to-end")
    ap.add_argument("--config", type=str, default="config/config.py")
    ap.add_argument("--splits", nargs="*", default=["eval"], help="splits to score: dev/eval")
    ap.add_argument("--calib-split", type=str, default="eval", choices=["dev", "eval"], help="which split to calibrate on")
    ap.add_argument("--k", type=int, default=128, help="number of KMeans buckets")
    ap.add_argument("--fit-sample-ratio", type=float, default=0.2, help="frame sampling ratio for PCA/KMeans fit")
    ap.add_argument("--alpha", type=float, default=1e-2, help="ridge alpha for linear-Gaussian fit")
    ap.add_argument("--var-floor", type=float, default=1e-4, help="variance floor for sigma")
    ap.add_argument("--trim", type=float, default=0.1, help="trim ratio for window-level NLL")
    ap.add_argument("--skip-unvoiced-ratio", type=float, default=0.8, help="skip windows with >= this unvoiced ratio")
    ap.add_argument("--out", type=str, default="results/c2s/score_c2s.jsonl")
    ap.add_argument("--score-batch", type=int, default=16, help="number of utterances per scoring batch (GPU utilization)")
    # force flags
    ap.add_argument("--force", action="store_true", help="force all stages")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--force-buckets", action="store_true")
    ap.add_argument("--force-lingauss", action="store_true")
    ap.add_argument("--force-score", action="store_true")
    ap.add_argument("--force-calib", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    # Prefer importing from project root so that 'config' is a package
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    try:
        from config.config import get_config  # type: ignore
        cfg = get_config()
    except Exception:
        # Fallback: load config module directly from file
        spec = importlib.util.spec_from_file_location("cfgmod", str(cfg_path))
        cfgmod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(cfgmod)  # type: ignore[arg-type]
        cfg = cfgmod.get_config()  # type: ignore[attr-defined]
    db_root = Path(cfg["database_path"])          # data/ASVspoof5
    feat_root = Path(cfg["feature_path"])         # data/ASVspoof5/features

    need_splits = set(["train"]) | set(args.splits)

    # 1) Ensure E/V caches
    need_extract = _need_extract_ev(feat_root, need_splits) or args.force or args.force_extract
    if need_extract:
        _extract_e_v(cfg, sorted(need_splits))
    else:
        print("[1/5] E/V caches present; skip extraction")

    # 2) Fit PCA+KMeans
    models_dir = Path("models/c2s")
    pca_path = models_dir / "pca64.npz"
    kmeans_path = models_dir / f"kmeans_k{int(args.k)}.npz"
    need_buckets = not (exist_nonempty(pca_path) and exist_nonempty(kmeans_path))
    if args.force or args.force_buckets:
        need_buckets = True
    if need_buckets:
        print(f"[2/5] Fitting PCA(64) and KMeans(K={args.k}) on train-bonafide (sample_ratio={args.fit_sample_ratio})")
        models_dir.mkdir(parents=True, exist_ok=True)
        _fit_buckets(cfg, models_dir, int(args.k), float(args.fit_sample_ratio))
    else:
        print("[2/5] PCA/KMeans present; skip bucket fitting")

    # 3) Fit linear–Gaussian per-bucket
    lg_path = models_dir / f"linGauss_k{int(args.k)}.npz"
    need_lg = not exist_nonempty(lg_path)
    if args.force or args.force_lingauss:
        need_lg = True
    if need_lg:
        print(f"[3/5] Fitting linear–Gaussian model (alpha={args.alpha}, var_floor={args.var_floor})")
        _fit_lin_gauss(cfg, pca_path, kmeans_path, lg_path, float(args.alpha), float(args.var_floor))
    else:
        print("[3/5] linGauss present; skip fitting")

    # 4) Score splits
    out_scores = Path(args.out)
    out_scores.parent.mkdir(parents=True, exist_ok=True)
    need_score = args.force or args.force_score or (not exist_nonempty(out_scores))
    if need_score:
        print(f"[4/5] Scoring splits {args.splits} with linGauss {lg_path.name}")
        _score_splits(cfg, lg_path, list(args.splits), out_scores, float(args.trim), float(args.skip_unvoiced_ratio))
    else:
        print("[4/5] Scores exist; skip scoring")

    # 5) Calibrate on chosen split
    out_cal = out_scores.with_name(out_scores.stem + "_calib.jsonl")
    need_cal = args.force or args.force_calib or (not exist_nonempty(out_cal))
    if need_cal:
        print(f"[5/5] Calibrating on split {args.calib_split}")
        _calibrate_and_eval(cfg, out_scores, out_cal, str(args.calib_split))
    else:
        print("[5/5] Calibrated scores exist; skip calibration")

    print("Done. Artifacts:")
    print(f"  PCA:        {pca_path}")
    print(f"  KMeans:     {kmeans_path}")
    print(f"  linGauss:   {lg_path}")
    print(f"  Raw scores: {out_scores}")
    print(f"  Calibrated: {out_cal}")

    # Metrics already printed by _calibrate_and_eval


if __name__ == "__main__":
    main()
