from __future__ import annotations

import random
from typing import List, Iterable, Iterator

from torch.utils.data import Sampler


class BlockBatchSampler(Sampler[List[int]]):
    """
    Yield batches of dataset indices with block-level shuffling and in-block sequential order.

    - index_blocks: list of blocks, each is a list of dataset indices that are contiguous on disk
                    (built by grouping records within the same shard and sorting by offset).
    - batch_size: number of samples per batch
    - drop_last: drop tail that does not fit a full batch within a block
    - seed: base seed; effective order changes by epoch via set_epoch(epoch)
    """

    def __init__(self, index_blocks: List[List[int]], batch_size: int, drop_last: bool, seed: int = 0) -> None:
        super().__init__(data_source=None)  # type: ignore[arg-type]
        self.index_blocks = [blk for blk in index_blocks if len(blk) > 0]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.index_blocks)))
        rng.shuffle(order)

        bs = self.batch_size
        for bi in order:
            block = self.index_blocks[bi]
            n = len(block)
            full = (n // bs) * bs
            # Full batches within the block
            for s in range(0, full, bs):
                yield block[s:s + bs]
            # Optional tail
            if not self.drop_last and full < n:
                yield block[full:n]

    def __len__(self) -> int:
        bs = self.batch_size
        total = 0
        for block in self.index_blocks:
            n = len(block)
            if self.drop_last:
                total += n // bs
            else:
                total += (n + bs - 1) // bs
        return total

