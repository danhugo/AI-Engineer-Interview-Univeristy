"""Paged KV cache with prefix sharing.

Why paging. A contiguous per-sequence cache has to be sized for the worst case,
so most of it sits unused, and two sequences can never share memory. Splitting
the cache into fixed blocks and keeping a per-sequence block_table means we
allocate only what is used, in any free slot. Fragmentation stops mattering.

Why prefix sharing. Requests repeat their openings — the same system prompt,
the same few-shot examples, the same chat history on every turn. Blocks are
content-addressed by a chained hash of their tokens, so an identical prefix
maps to the blocks that already hold its K/V. Those tokens then need no
attention compute at all.

Two rules keep sharing safe:

  hashes are published only after the K/V is written. Otherwise a sequence in
  the same batch could "hit" a block whose contents do not exist yet and read
  garbage. allocate() records pending hashes; commit() publishes them once the
  forward pass has run.

  a hit is confirmed by comparing tokens, not just the hash. A 64-bit
  collision would otherwise silently serve the wrong K/V.
"""

import hashlib
from collections import deque
from dataclasses import dataclass, field

import torch

from engine.sequence import Sequence

NO_HASH = -1


def compute_hash(token_ids: list[int], prefix_hash: int = NO_HASH) -> int:
    """Chained hash of a block's tokens.

    Chaining matters: block 2 of "A B C" must not match block 2 of "X B C".
    Folding the previous block's hash in makes the hash identify the whole
    prefix up to this block, not just its own tokens.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(prefix_hash.to_bytes(8, "little", signed=True))
    for t in token_ids:
        h.update(t.to_bytes(4, "little"))
    return int.from_bytes(h.digest(), "little", signed=True)


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    hash: int = NO_HASH
    token_ids: list[int] = field(default_factory=list)


class BlockManager:
    """Owns the KV cache tensors, the free list, and the prefix index."""

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 num_blocks: int = 64, block_size: int = 256,
                 dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
                 enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        # One (k, v) pair per layer. Shape is what flash-attn's paged kernels
        # want: (num_blocks, block_size, num_kv_heads, head_dim).
        #
        # One extra block past the usable pool is the sink. CUDA graphs need a
        # fixed batch size, so a decode step with fewer sequences is padded —
        # and those padded rows still execute a K/V write. Aiming them at the
        # sink keeps them from corrupting a real sequence's blocks. Nothing
        # ever reads it.
        self.sink_block_id = num_blocks
        self.sink_slot = num_blocks * block_size
        shape = (num_blocks + 1, block_size, num_kv_heads, head_dim)
        self.k_caches = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v_caches = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]

        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = deque(range(num_blocks))
        self.hash_to_block_id: dict[int, int] = {}

        self.num_hits = 0     # tokens served from the prefix cache
        self.num_misses = 0   # tokens that had to be computed

    # ---------------- accounting ----------------

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def bytes_per_token(self) -> int:
        k = self.k_caches[0]
        per_layer = 2 * k.shape[2] * k.shape[3] * k.element_size()  # k and v
        return per_layer * len(self.k_caches)

    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def hit_rate(self) -> float:
        total = self.num_hits + self.num_misses
        return self.num_hits / total if total else 0.0

    # ---------------- low-level block moves ----------------

    def _take_free(self) -> int:
        """Claim a free block, dropping any prefix entry that pointed at it.

        A block can sit in the free list while still being indexed by hash —
        that is the cache holding on to it in case someone wants the prefix
        again. Reusing it for new data means that entry must go. This is the
        eviction policy: FIFO over the free list.
        """
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        if block.hash != NO_HASH:
            self.hash_to_block_id.pop(block.hash, None)
            block.hash = NO_HASH
            block.token_ids = []
        block.ref_count = 1
        return block_id

    def _acquire(self, block_id: int) -> None:
        """Take a reference to an existing block, un-freeing it if needed."""
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self.free_block_ids.remove(block_id)  # small pools; O(n) is fine
        block.ref_count += 1

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            # Keep the hash: the block stays reusable as a prefix until someone
            # claims it for new data.
            self.free_block_ids.append(block_id)

    # ---------------- allocation ----------------

    def can_allocate(self, seq: Sequence) -> bool:
        need = seq.num_blocks_needed(self.block_size) - len(seq.block_table)
        return need <= self.num_free_blocks

    def allocate(self, seq: Sequence) -> None:
        """Give seq blocks for all its tokens, reusing any cached prefix.

        Sets seq.num_cached to how many leading tokens already have K/V in the
        cache. Those are skipped by the prefill forward pass.
        """
        if seq.block_table:
            self._grow(seq)
            return

        bs = self.block_size
        n = len(seq)
        num_full = n // bs                  # blocks with all their tokens known
        num_total = seq.num_blocks_needed(bs)

        table: list[int] = []
        pending: list[tuple[int, int, list[int]]] = []
        prefix_hash = NO_HASH
        cached = 0

        # Reuse the longest published prefix. It has to be contiguous from the
        # start, so the first miss ends the search.
        if self.enable_prefix_caching:
            for i in range(num_full):
                tokens = seq.token_ids[i * bs:(i + 1) * bs]
                h = compute_hash(tokens, prefix_hash)
                block_id = self.hash_to_block_id.get(h)
                if block_id is None or self.blocks[block_id].token_ids != tokens:
                    break
                self._acquire(block_id)
                table.append(block_id)
                prefix_hash = h
                cached += bs

        # Fresh blocks for the rest.
        for i in range(len(table), num_total):
            block_id = self._take_free()
            table.append(block_id)
            if i < num_full and self.enable_prefix_caching:
                tokens = seq.token_ids[i * bs:(i + 1) * bs]
                prefix_hash = compute_hash(tokens, prefix_hash)
                pending.append((block_id, prefix_hash, tokens))

        # A whole-prompt hit would leave nothing to run the forward pass on.
        # Recompute the last cached block; it writes back identical K/V.
        if cached == n and cached > 0:
            cached -= bs

        seq.block_table = table
        seq.num_cached = cached
        seq.pending_hashes = pending
        self.num_hits += cached
        self.num_misses += n - cached

    def _grow(self, seq: Sequence) -> None:
        """Decode-time growth: add a block if the next token needs one.

        Also queues a hash for any block that just filled up. Queued, not
        published — the K/V for that last token is written by the forward pass
        that follows.
        """
        bs = self.block_size
        need = seq.num_blocks_needed(bs) - len(seq.block_table)
        for _ in range(need):
            seq.block_table.append(self._take_free())

        if not self.enable_prefix_caching:
            return
        n = len(seq)
        if n % bs == 0 and n > 0:
            idx = n // bs - 1
            block_id = seq.block_table[idx]
            if self.blocks[block_id].hash == NO_HASH:
                tokens = seq.token_ids[idx * bs:n]
                prev = self.blocks[seq.block_table[idx - 1]].hash if idx else NO_HASH
                seq.pending_hashes = [(block_id, compute_hash(tokens, prev), tokens)]

    def commit(self, seq: Sequence) -> None:
        """Publish hashes for blocks whose K/V is now written.

        Called after the forward pass. Before this point the blocks hold
        nothing, and a sequence that "hit" them would read zeros.
        """
        for block_id, h, tokens in seq.pending_hashes:
            block = self.blocks[block_id]
            block.hash = h
            block.token_ids = list(tokens)
            self.hash_to_block_id[h] = block_id
        seq.pending_hashes = []

    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            self._release(block_id)
        seq.block_table.clear()
        seq.pending_hashes = []
        seq.num_cached = 0

    # ---------------- addressing ----------------

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


def build_manager_for(model, num_blocks: int = 64, block_size: int = 256,
                      enable_prefix_caching: bool = True) -> BlockManager:
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
        enable_prefix_caching=enable_prefix_caching,
    )
    attach_kv_cache(model, manager)
    return manager
