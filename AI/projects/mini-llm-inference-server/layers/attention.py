"""Attention that reads and writes a paged KV cache.

Two paths, because prefill and decode are different shapes of problem:

  prefill  the whole prompt arrives at once, so we hold every K/V for this
           sequence in hand. Write them to the cache, then attend over them
           directly — no need to read back through the block table.
  decode   one new token. Its K/V goes into the cache, then we attend over
           everything cached so far, reached through the block table.

Stage 2 stores K/V with plain torch indexing. A Triton kernel replaces that
later; correctness first.
"""

import torch

from utils.context import Context

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except ImportError:  # pragma: no cover - depends on the machine
    flash_attn_func = flash_attn_with_kvcache = None


def store_kv(k: torch.Tensor, v: torch.Tensor,
             k_cache: torch.Tensor, v_cache: torch.Tensor,
             slot_mapping: torch.Tensor) -> None:
    """Scatter this step's K/V into their cache slots.

    k, v          (num_tokens, num_kv_heads, head_dim)
    caches        (num_blocks, block_size, num_kv_heads, head_dim)
    slot_mapping  (num_tokens,) flat slot = block_id * block_size + offset

    Flattening the first two cache dims turns the block layout into one long
    array of token slots, so a single index assignment does the scatter.
    """
    num_heads, head_dim = k.shape[-2:]
    k_cache.view(-1, num_heads, head_dim)[slot_mapping] = k
    v_cache.view(-1, num_heads, head_dim)[slot_mapping] = v


def paged_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 k_cache: torch.Tensor, v_cache: torch.Tensor,
                 ctx: Context) -> torch.Tensor:
    """Causal GQA against the paged cache. q/k/v are (batch, seq, heads, dim)."""
    b, s = q.shape[:2]
    store_kv(k.reshape(-1, *k.shape[-2:]), v.reshape(-1, *v.shape[-2:]),
             k_cache, v_cache, ctx.slot_mapping)

    if ctx.is_prefill:
        # Every K/V this sequence needs was computed in this same call.
        return flash_attn_func(q, k, v, causal=True)

    # Decode: read the whole history back out of the blocks.
    return flash_attn_with_kvcache(
        q, k_cache, v_cache,
        cache_seqlens=ctx.cache_seqlens,
        block_table=ctx.block_table,
        causal=True,
    )
