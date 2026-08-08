"""Datasets for transformer training.

Two cases, same Dataset interface (torch.utils.data.Dataset):

  Case 1 — InMemoryDataset: whole corpus fits in RAM.
           __getitem__ is an O(1) slice (microseconds). num_workers=0
           is usually fine; prefetch barely matters. Simplest.

  Case 2 — StreamingDataset: too big for RAM. Each __getitem__ reads
           one window from a disk-backed tensor. RAM stays flat no
           matter how big the corpus is. num_workers>1 pays off here:
           reads (and any decode) run in parallel, hidden behind GPU
           compute.

Both return (input_ids, target_ids) where target = input shifted by 1
(next-token prediction). Both are 1D LongTensors of length seq_len.

Uses torch's real Dataset and DataLoader. (data.py is a study-only
from-scratch reimplementation of the same idea.)
"""
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ===========================================================================
# Case 1: whole corpus fits in RAM
# ===========================================================================
class InMemoryDataset(Dataset):
    """All token IDs already in memory as one big 1D tensor.

    Use when the corpus is small enough to hold in RAM (a few million
    tokens). __getitem__ is an O(1) slice, so num_workers=0 is usually
    fine and prefetch barely matters. Build once at startup, reuse
    across epochs.
    """

    def __init__(self, token_ids: Tensor, seq_len: int):
        self.tokens = token_ids.long().view(-1)
        self.seq_len = seq_len
        # number of non-overlapping windows. -1 because we also need a
        # +1 shifted target; drop the last partial chunk so every
        # example is exactly seq_len.
        self.n = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        start = i * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]              # input:  [start .. end-1]
        y = self.tokens[start + 1:end + 1]      # target: [start+1 .. end]
        return x, y


# ===========================================================================
# Case 2: too big for RAM — stream one window per __getitem__
# ===========================================================================
class StreamingDataset(Dataset):
    """Token IDs live in a .pt file; each __getitem__ reads one window.

    Uses torch's own mmap (torch.load(mmap=True)) so init is O(1) and
    RAM stays flat no matter how big the corpus is. The file is mapped
    into the process's address space but pages are only fetched from
    disk on demand — slicing a window reads just those pages, not the
    whole file.

    Bonus: multiple worker processes mmap the same file → the OS maps
    them all to the same physical pages (copy-on-write). No N copies of
    the data — that's the RAM-avoidance trick behind torch's workers.

    __getitem__ does a real read each call (page fault on cold pages),
    so num_workers>1 pays off: reads run in parallel across processes,
    hidden behind GPU compute.

    File format: a torch .pt file (1D LongTensor) saved with torch.save.
    """

    def __init__(self, path: str, seq_len: int, vocab_size: int = 1000):
        self.path = path
        self.seq_len = seq_len
        # mmap=True: map the .pt file instead of loading it. Pages are
        # fetched from disk lazily as we slice. weights_only=True keeps it
        # safe (no arbitrary unpickling).
        self.tokens = torch.load(
            path, map_location="cpu", mmap=True, weights_only=True
        ).long().view(-1)
        self.n = (len(self.tokens) - 1) // self.seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        start = i * self.seq_len
        end = start + self.seq_len
        # .clone() because a slice of an mmap'd tensor is a view into the
        # mapping; we want a standalone tensor so collate can stack it
        # without repeatedly page-faulting the same region.
        x = self.tokens[start:end].clone()
        y = self.tokens[start + 1:end + 1].clone()
        return x, y
