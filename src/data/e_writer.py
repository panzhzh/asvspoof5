"""
Ragged memmap writer for excitation features E (float16) and voiced mask V (uint8),
organized by split: data/ASVspoof5/features/{E|V}/{train,dev,eval}/

Each split directory contains shards data_XXX.npy and index.jsonl with entries:
{utt_id, shard, offset_elems, elem_count, T, D, L=1, sr=16000,
 win_ms=25, stride_ms=20, dtype, layout:"TD", version:"E.v1.1"}

Note: E and V maintain their own indices (different dtype/D and element counts),
but share utt_id and T for alignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


@dataclass
class ShardSpec:
    root: Path            # split root dir (e.g., features/E/train)
    dtype: np.dtype       # np.float16 for E, np.uint8 for V
    target_gb: float = 2.0
    prefix: str = "data_"
    version: str = "E.v1.1"


class RaggedMemmapWriter:
    def __init__(self, spec: ShardSpec, meta_common: dict):
        self.spec = spec
        self.meta_common = meta_common.copy()
        self.root = spec.root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_fp = (self.root / "index.jsonl").open("w")
        self.dtype = spec.dtype
        self.bytes_per_elem = np.dtype(self.dtype).itemsize
        # capacity in elements
        self.capacity_elems = int(spec.target_gb * (1024 ** 3) / self.bytes_per_elem)
        self.shard_id = -1
        self.offset = 0
        self.mm = None
        self._new_shard()

    def _new_shard(self):
        if self.mm is not None:
            self.mm.flush()
            del self.mm
        self.shard_id += 1
        self.offset = 0
        shard_path = self.root / f"{self.spec.prefix}{self.shard_id:03d}.npy"
        self.mm = np.lib.format.open_memmap(
            shard_path, mode='w+', shape=(self.capacity_elems,), dtype=self.dtype
        )

    def write_sample(self, utt_id: str, flat_arr: np.ndarray, T: int, D: int):
        elem_count = int(flat_arr.size)
        if self.offset + elem_count > self.capacity_elems:
            # trim previous shard to actual size
            shard_path = self.root / f"{self.spec.prefix}{self.shard_id:03d}.npy"
            used = self.offset
            self.mm.flush()
            if used < self.capacity_elems:
                tmp = np.lib.format.open_memmap(
                    shard_path.with_suffix('.tmp.npy'), mode='w+', shape=(used,), dtype=self.dtype
                )
                tmp[:] = self.mm[:used]
                tmp.flush()
                del tmp
                shard_path.unlink()
                shard_path.with_suffix('.tmp.npy').rename(shard_path)
            self._new_shard()
        # write
        self.mm[self.offset:self.offset + elem_count] = flat_arr
        # index record
        rec = {
            **self.meta_common,
            "utt_id": utt_id,
            "shard": f"{self.spec.prefix}{self.shard_id:03d}.npy",
            "offset_elems": int(self.offset),
            "elem_count": int(elem_count),
            "T": int(T),
            "D": int(D),
            "dtype": str(self.dtype).split(".")[-1],
            "version": self.spec.version,
        }
        self.index_fp.write(json.dumps(rec) + "\n")
        self.offset += elem_count

    def close(self):
        if self.mm is not None:
            # final trim
            shard_path = self.root / f"{self.spec.prefix}{self.shard_id:03d}.npy"
            used = self.offset
            self.mm.flush()
            if used < self.capacity_elems:
                tmp = np.lib.format.open_memmap(
                    shard_path.with_suffix('.tmp.npy'), mode='w+', shape=(used,), dtype=self.dtype
                )
                tmp[:] = self.mm[:used]
                tmp.flush()
                del tmp
                shard_path.unlink()
                shard_path.with_suffix('.tmp.npy').rename(shard_path)
            del self.mm
            self.mm = None
        if not self.index_fp.closed:
            self.index_fp.close()

