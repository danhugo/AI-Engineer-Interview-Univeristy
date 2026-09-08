"""Load HuggingFace weights, slicing them for tensor parallelism.

Our module and parameter names match HuggingFace exactly, so with TP off this
is just load_state_dict. With TP on, each rank takes its slice: the shard
dimension comes from the layer class (ColumnParallelLinear splits dim 0,
RowParallelLinear splits dim 1) rather than from a name lookup table, so
nothing has to be kept in sync by hand.
"""

import torch
from torch import nn

from layers.linear import ColumnParallelLinear, RowParallelLinear
from utils import parallel


def _slice(full: torch.Tensor, dim: int, rank: int, world: int) -> torch.Tensor:
    size = full.shape[dim] // world
    return full.narrow(dim, rank * size, size)


def load_weights(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Copy weights into model, sharding parallel layers for this rank."""
    rank, world = parallel.rank(), parallel.world_size()

    # Which parameters belong to a sharded layer, and along which dim.
    shard_dim: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, ColumnParallelLinear):
            shard_dim[f"{name}.weight"] = 0
            shard_dim[f"{name}.bias"] = 0
        elif isinstance(module, RowParallelLinear):
            shard_dim[f"{name}.weight"] = 1
            # A row-parallel bias is not sharded; only rank 0 holds it.

    own = dict(model.named_parameters())
    missing, unexpected = [], []

    for name, param in own.items():
        if name not in state_dict:
            missing.append(name)
            continue
        full = state_dict[name]
        dim = shard_dim.get(name)
        want = _slice(full, dim, rank, world) if (dim is not None and world > 1) else full
        assert param.shape == want.shape, \
            f"{name}: model wants {tuple(param.shape)}, weights give {tuple(want.shape)}"
        param.data.copy_(want)

    for name in state_dict:
        if name not in own:
            unexpected.append(name)

    # Tied embeddings mean lm_head.weight is absent from the checkpoint; the
    # model already aliases it, so that is expected.
    tied = getattr(model.config, "tie_word_embeddings", False)
    missing = [m for m in missing if not (tied and m == "lm_head.weight")]
    assert not missing, f"missing weights: {missing[:5]}"
    if unexpected:
        parallel.log(f"[loader] ignoring {len(unexpected)} unexpected keys, "
                     f"e.g. {unexpected[:3]}")
