#!/usr/bin/env python3
"""
Extract excitation features E (32-d, float16) and voiced mask V (uint8) for
train/dev/eval into ragged memmap shards with index.jsonl.

Usage:
  python scripts/c2s_extract_E.py [--splits train dev eval] [--shard-gb 2.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa

import sys
from pathlib import Path as _Path
# Ensure project root on sys.path (so `import config.config` works)
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from config.config import c2s as C2S_CFG
from src.features.excitation import extract_excitation_gpu, EConfig
from src.data.e_writer import RaggedMemmapWriter, ShardSpec


def get_file_list(tsv_path: Path) -> list[str]:
    ids = []
    with tsv_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                ids.append(parts[1])
            else:
                ids.append(parts[0])
    return ids


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = librosa.to_mono(wav.T)
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    wav = np.clip(wav.astype(np.float32), -1.0, 1.0)
    return wav


def main():
    parser = argparse.ArgumentParser(description="C2S: extract E/V for splits")
    parser.add_argument("--splits", nargs="*", default=C2S_CFG.get("extract_splits", ["train", "dev", "eval"]))
    parser.add_argument("--shard-gb", type=float, default=C2S_CFG.get("shard_gb", 2.0))
    args = parser.parse_args()

    root = Path(C2S_CFG["out_root"]).resolve()
    db_root = Path("data/ASVspoof5").resolve()

    split_map = {
        "train": (db_root / "ASVspoof5.train.tsv", db_root / "flac_T"),
        "dev": (db_root / "ASVspoof5.dev.track_1.tsv", db_root / "flac_D"),
        "eval": (db_root / "ASVspoof5.eval.track_1.tsv", db_root / "flac_E"),
    }

    # Common meta
    e_cfg = EConfig(
        sr=16000,
        n_fft=512,
        win_length=int(C2S_CFG.get("e_win_ms", 25) / 1000 * 16000) if isinstance(C2S_CFG.get("e_win_ms", 25), float) else 400,
        hop_length=int(C2S_CFG.get("e_stride_ms", 20) / 1000 * 16000) if isinstance(C2S_CFG.get("e_stride_ms", 20), float) else 320,
        center=True,
    )
    meta_common = {
        "L": 1,
        "sr": e_cfg.sr,
        "win_ms": 25,
        "stride_ms": 20,
        "layout": "TD",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for split in args.splits:
        if split not in split_map:
            print(f"Skip unknown split: {split}")
            continue
        tsv_path, wav_dir = split_map[split]
        if not wav_dir.exists():
            print(f"[WARN] audio dir missing for {split}: {wav_dir}")
            continue
        ids = get_file_list(tsv_path)
        print(f"Split {split}: {len(ids)} files")

        # Writers
        e_spec = ShardSpec(root=root / "E" / split, dtype=np.float16, target_gb=args.shard_gb)
        v_spec = ShardSpec(root=root / "V" / split, dtype=np.uint8, target_gb=args.shard_gb)
        ew = RaggedMemmapWriter(e_spec, meta_common)
        vw = RaggedMemmapWriter(v_spec, meta_common)

        # Running mean/std for E (per-dim)
        mean = None
        M2 = None
        count = 0

        for uid in ids:
            wav_path = wav_dir / f"{uid}.flac"
            if not wav_path.exists():
                # try wav fallback
                wav_path = wav_dir / f"{uid}.wav"
                if not wav_path.exists():
                    print(f"[MISS] {wav_path}")
                    continue
            wav = load_wav(wav_path)
            wav_t = torch.from_numpy(wav).to(device)
            with torch.no_grad():
                E, V = extract_excitation_gpu(wav_t, e_cfg)  # E: [T,32], V: [T]
            E = E.cpu().numpy().astype(np.float32)
            V = V.cpu().numpy().astype(np.uint8)

            T = int(E.shape[0])
            De = int(E.shape[1])
            # update running stats
            if mean is None:
                mean = E.sum(axis=0)
                M2 = (E ** 2).sum(axis=0)
                count = T
            else:
                mean += E.sum(axis=0)
                M2 += (E ** 2).sum(axis=0)
                count += T

            # cast & write
            ew.write_sample(uid, E.astype(np.float16).ravel(), T=T, D=De)
            vw.write_sample(uid, V.ravel(), T=T, D=1)

        ew.close()
        vw.close()

        # Save per-dim mean/std
        if count > 0:
            mu = (mean / count).astype(np.float32)
            var = (M2 / count) - mu ** 2
            var = np.maximum(var, 1e-8)
            std = np.sqrt(var).astype(np.float32)
            stats = {"mean": mu.tolist(), "std": std.tolist(), "count_frames": int(count)}
            (e_spec.root / "stats.json").write_text(json.dumps(stats))
            print(f"Saved stats for {split}: frames={count}")

    print("Done.")


if __name__ == "__main__":
    main()
