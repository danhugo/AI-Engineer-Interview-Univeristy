"""Synthetic seq2seq task: reverse a digit sequence.

No download, trains in seconds. This is the harness for proving the training
loop is correct before spending time on real data.

Why reversal and not copy: copying can be solved by the residual stream alone,
so a broken attention mask still scores well. Reversing requires position i of
the output to attend to position n-i of the input, so it only works if
cross-attention and the causal mask are both right.

    src:  3 1 4 1 5
    tgt:  5 1 4 1 3

Vocabulary is tiny and fixed:

    0 <pad>   1 <unk>   2 <bos>   3 <eos>   4..13 digits 0-9
"""
import torch
from torch import Tensor
from torch.utils.data import Dataset

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
DIGIT_OFFSET = 4
TOY_VOCAB_SIZE = DIGIT_OFFSET + 10  # 14


class ReverseDigitsDataset(Dataset):
    """Random digit sequences paired with their reverse.

    Args:
        n_samples: how many pairs
        min_len / max_len: sequence length range, sampled per example
        seed: fixed so train and eval splits are reproducible

    Returns (src, tgt) as 1D LongTensors. src has no special tokens; tgt is
    wrapped as <bos> ... <eos>. The train/eval shift happens in collate.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        min_len: int = 4,
        max_len: int = 10,
        seed: int = 0,
    ):
        if min_len < 1 or max_len < min_len:
            raise ValueError("need 1 <= min_len <= max_len")
        self.n_samples = n_samples
        self.min_len = min_len
        self.max_len = max_len
        # Generate everything up front with a fixed generator. Doing it lazily
        # in __getitem__ would give different data to each DataLoader worker.
        g = torch.Generator().manual_seed(seed)
        self.pairs = [self._make(g) for _ in range(n_samples)]

    def _make(self, g: torch.Generator) -> tuple[Tensor, Tensor]:
        length = int(torch.randint(self.min_len, self.max_len + 1, (1,), generator=g))
        digits = torch.randint(0, 10, (length,), generator=g) + DIGIT_OFFSET
        src = digits
        tgt = torch.cat([
            torch.tensor([BOS_ID]),
            digits.flip(0),
            torch.tensor([EOS_ID]),
        ])
        return src, tgt

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.pairs[i]


def decode_toy(ids: list[int] | Tensor) -> str:
    """Turn token IDs back into a readable digit string."""
    if isinstance(ids, Tensor):
        ids = ids.tolist()
    out = []
    for i in ids:
        if i in (PAD_ID, BOS_ID):
            continue
        if i == EOS_ID:
            break
        out.append(str(i - DIGIT_OFFSET) if i >= DIGIT_OFFSET else "?")
    return "".join(out)
