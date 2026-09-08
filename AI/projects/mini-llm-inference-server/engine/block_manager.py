"""Paged KV cache: a pool of fixed-size blocks handed out to sequences.

Why paging. A contiguous per-sequence cache has to be sized for the worst case,
so most of it sits unused, and two sequences can never share memory. Splitting
the cache into fixed blocks and keeping a per-sequence block_table means we
allocate only what is used, in any free slot. Fragmentation stops mattering.

Stage 2 does allocation and freeing. Prefix sharing (ref counts, hashing) is
stage 4.
"""

from collections import deque

import torch

from engine.sequence import Sequence


class BlockManager:
    """Owns the KV cache tensors and the free-block list."""

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 num_blocks: int = 64, block_size: int = 256,
                 dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # One (k, v) pair per layer. Shape is what flash-attn's paged kernels
        # want: (num_blocks, block_size, num_kv_heads, head_dim).
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_caches = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v_caches = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]

        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def bytes_per_token(self) -> int:
        k = self.k_caches[0]
        per_layer = 2 * k.shape[2] * k.shape[3] * k.element_size()  # k and v
        return per_layer * len(self.k_caches)

    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def can_allocate(self, seq: Sequence) -> bool:
        need = seq.num_blocks_needed(self.block_size) - len(seq.block_table)
        return need <= self.num_free_blocks

    def allocate(self, seq: Sequence) -> None:
        """Give seq enough blocks to hold all its tokens."""
        need = seq.num_blocks_needed(self.block_size) - len(seq.block_table)
        if need > self.num_free_blocks:
            raise RuntimeError(
                f"out of KV blocks: need {need}, free {self.num_free_blocks}. "
                f"Capacity is {self.capacity_tokens()} tokens."
            )
        for _ in range(need):
            block_id = self.free_block_ids.popleft()
            self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        for block_id in seq.block_table:
            self.used_block_ids.discard(block_id)
            self.free_block_ids.append(block_id)
        seq.block_table.clear()
        seq.num_cached = 0

    def slot_mapping(self, seq: Sequence, start: int, end: int) -> list[int]:
        """Flat cache slots for tokens [start, end) of seq.

        Slot = block_id * block_size + offset_within_block. The attention layer
        writes K/V straight into these positions.
        """
        slots = []
        for pos in range(start, end):
            block_id = seq.block_table[pos // self.block_size]
            slots.append(block_id * self.block_size + pos % self.block_size)
        return slots


def attach_kv_cache(model, manager: BlockManager) -> None:
    """Point each attention layer at its slice of the cache.

    The layers then take the paged path instead of recomputing K/V every step.
    """
    layers = model.model.layers
    assert len(layers) == len(manager.k_caches), \
        f"model has {len(layers)} layers, cache has {len(manager.k_caches)}"
    for layer, k, v in zip(layers, manager.k_caches, manager.v_caches):
        layer.self_attn.k_cache = k
        layer.self_attn.v_cache = v


def build_manager_for(model, num_blocks: int = 64, block_size: int = 256) -> BlockManager:
    """Size a BlockManager from the model's own config, and wire it up."""
    cfg = model.config
    attn = model.model.layers[0].self_attn
    manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=attn.num_kv_heads,
        head_dim=attn.head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    attach_kv_cache(model, manager)
    return manager
