"""
Joint C+E+V loader for C²S pipeline with 4s/2s sliding windows.

Reads:
- Content C from ragged_memmap CSR under feature_root/{train,dev,eval}
- Excitation E (32-d) from features/E/{split}
- Voiced mask V (1-d, {0,1}) from features/V/{split}

Returns per-item:
- C_TD: (T, 288) float32  (mean over layers if input is [L,T,D])
- E_TD: (T, 32)  float32
- V_T:  (T,)     uint8
- segments: list[(start,end)] frame indices for 4s win, 2s hop
- utt_id: str

Notes:
- This module is for C²S scoring/train-time utilities. It does not implement
  block-aggregated I/O; use it in conjunction with moderate batch sizes or
  in offline steps (bucket/model fitting, scoring).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .datasets import FeatureLoader  # reuse existing loader for C


@dataclass
class RaggedIndex:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.index = self.root / "index.jsonl"
        self.records: dict[str, dict] = {}
        self.mm_cache: dict[str, np.memmap] = {}
        with self.index.open() as f:
            for line in f:
                r = json_loads(line)
                self.records[r["utt_id"]] = r

    def _mm(self, shard: str):
        mm = self.mm_cache.get(shard)
        if mm is None:
            mm = np.lib.format.open_memmap(self.root / shard, mode='r')
            self.mm_cache[shard] = mm
        return mm

    def get(self, uid: str) -> np.ndarray:
        r = self.records[uid]
        mm = self._mm(r["shard"])
        off = int(r["offset_elems"])
        cnt = int(r["elem_count"])
        T = int(r.get("T", r.get("real_len")))
        D = int(r.get("D", max(1, cnt // max(1, T))))
        flat = mm[off:off+cnt]
        arr = np.asarray(flat).reshape(T, D)
        return arr


def json_loads(line: str) -> dict:
    import json
    # be robust to dtype strings like "float16'>"
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # quick hack: remove stray "'>" from dtype fields
        fixed = line.replace("\"float16'\>", '"float16"').replace("\"uint8'\>", '"uint8"')
        return json.loads(fixed)


def make_segments(n_frames: int, win_frames: int = 200, hop_frames: int = 100) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    if n_frames <= 0:
        return segs
    s = 0
    while s + 1 < n_frames:
        e = min(n_frames, s + win_frames)
        segs.append((s, e))
        if e >= n_frames:
            break
        s = s + hop_frames
    return segs


class C2SDataset:
    def __init__(self,
                 split: str,
                 feature_root: Path,
                 e_root: Path,
                 v_root: Path,
                 utt_ids: list[str],
                 win_frames: int = 200,
                 hop_frames: int = 100,
                 ) -> None:
        self.split = str(split)
        self.feature_root = Path(feature_root)
        self.e_index = RaggedIndex(Path(e_root))
        self.v_index = RaggedIndex(Path(v_root))
        self.c_loader = FeatureLoader(Path(feature_root))
        self.utt_ids = list(utt_ids)
        self.win_frames = int(win_frames)
        self.hop_frames = int(hop_frames)

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx: int):
        uid = self.utt_ids[idx]
        # C: [L,T,288] -> mean over L -> [T,288]
        C_ltd = self.c_loader.get_features(uid)
        if C_ltd.ndim != 3:
            raise ValueError(f"Unexpected C shape for {uid}: {C_ltd.shape}")
        C_td = C_ltd.mean(axis=0).astype(np.float32)
        # E: [T,32]
        E_td = self.e_index.get(uid).astype(np.float32)
        # V: [T,1] or [T]
        V_t = self.v_index.get(uid).astype(np.uint8).reshape(-1)
        # Align lengths (allow +-1)
        T = min(C_td.shape[0], E_td.shape[0], V_t.shape[0])
        C_td = C_td[:T]
        E_td = E_td[:T]
        V_t = V_t[:T]
        segs = make_segments(T, self.win_frames, self.hop_frames)
        return C_td, E_td, V_t, segs, uid

