"""Sharded linear layers for tensor parallelism.

A matmul can be split two ways, and a transformer block uses both so that only
one all-reduce is needed per pair instead of one per layer.

Column parallel — split the OUTPUT dimension:

    Y = X @ [W1 | W2]  ->  rank0 computes X @ W1, rank1 computes X @ W2

    Each rank holds a different slice of the output and no communication is
    needed. Used for q/k/v (each rank owns whole attention heads) and for
    gate/up (each rank owns a slice of the hidden dimension).

Row parallel — split the INPUT dimension:

    Y = [X1 | X2] @ [W1]  ->  rank0: X1 @ W1, rank1: X2 @ W2, then SUM
                        [W2]

    Each rank produces a partial sum of the full output, so one all-reduce
    finishes it. Used for o_proj and down_proj.

Chaining column then row is the trick: q/k/v hand their sliced output straight
into attention and then into o_proj's sliced input, so the intermediate never
has to be gathered. One all-reduce per pair.
"""

import torch
import torch.nn.functional as F
from torch import nn

from utils import parallel


class ColumnParallelLinear(nn.Module):
    """Holds rows [rank*out_per_rank : (rank+1)*out_per_rank] of the weight."""

    shard_dim = 0  # of the full (out_features, in_features) weight

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        world = parallel.world_size()
        assert out_features % world == 0, \
            f"out_features {out_features} not divisible by TP size {world}"
        self.in_features = in_features
        self.out_features_full = out_features
        self.out_features = out_features // world

        self.weight = nn.Parameter(torch.empty(self.out_features, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Holds columns [rank*in_per_rank : (rank+1)*in_per_rank] of the weight."""

    shard_dim = 1

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        world = parallel.world_size()
        assert in_features % world == 0, \
            f"in_features {in_features} not divisible by TP size {world}"
        self.in_features_full = in_features
        self.in_features = in_features // world
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features))
        # A replicated bias would be added once per rank, so only rank 0 keeps
        # it. Qwen3 has no biases here, but the rule matters if that changes.
        keep_bias = bias and parallel.rank() == 0
        self.bias = nn.Parameter(torch.empty(out_features)) if keep_bias else None

    def forward(self, x):
        # Each rank's slice gives a partial sum; all_reduce completes it.
        return parallel.all_reduce(F.linear(x, self.weight, self.bias))


def linear(in_features: int, out_features: int, bias: bool = False,
           kind: str = "plain") -> nn.Module:
    """Pick a linear layer. Falls back to nn.Linear when TP is off."""
    if parallel.world_size() == 1 or kind == "plain":
        return nn.Linear(in_features, out_features, bias=bias)
    if kind == "column":
        return ColumnParallelLinear(in_features, out_features, bias)
    if kind == "row":
        return RowParallelLinear(in_features, out_features, bias)
    raise ValueError(f"unknown kind {kind!r}")
