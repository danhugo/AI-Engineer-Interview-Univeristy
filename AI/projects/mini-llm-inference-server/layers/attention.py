"""Attention over a paged KV cache, batched.

All tensors are flattened to (num_tokens, heads, head_dim) — no batch dim, no
padding. Sequence boundaries come from cu_seqlens. Three paths:

  prefill (paged)  many prompts of different lengths concatenated. Write every
                   K/V to the cache, then one varlen kernel call attends within
                   each sequence, using cu_seqlens to keep them separate.
  decode (paged)   one new token per sequence. Write its K/V, then read the
                   whole history back through each sequence's block table.
  no cache         stage-1 path, single sequence, recompute everything.

Stage 3 stores K/V with plain torch indexing. A Triton kernel replaces that
later; correctness first.
"""

import torch

from utils.context import Context

try:
    from flash_attn import (
        flash_attn_func,
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )
except ImportError:  # pragma: no cover - depends on the machine
    flash_attn_func = flash_attn_varlen_func = flash_attn_with_kvcache = None


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


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           k_cache: torch.Tensor | None, v_cache: torch.Tensor | None,
           ctx: Context) -> torch.Tensor:
    """Causal GQA. q/k/v are (num_tokens, heads, head_dim); returns the same."""
    if k_cache is None or not ctx.active:
        # Stage-1 path: one sequence, no cache. flash_attn_func wants a batch
        # dim; SDPA covers fp32 and machines without flash-attn.
        return _uncached(q, k, v)

    store_kv(k, v, k_cache, v_cache, ctx.slot_mapping)

    if ctx.is_prefill:
        # Every K/V needed this step was just computed. cu_seqlens keeps the
        # concatenated sequences from attending across each other.
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_k=ctx.cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            causal=True,
        )

    # Decode: one query token per sequence, history read via the block table.
    o = flash_attn_with_kvcache(
        q.unsqueeze(1),  # (batch, seqlen_q=1, heads, dim)
        k_cache, v_cache,
        cache_seqlens=ctx.cache_seqlens,
        block_table=ctx.block_table,
        causal=True,
    )
    return o.squeeze(1)


def _uncached(q, k, v):
    if flash_attn_func is not None and q.dtype in (torch.float16, torch.bfloat16):
        return flash_attn_func(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                               causal=True).squeeze(0)
    # SDPA wants (batch, heads, seq, dim); enable_gqa broadcasts the kv heads.
    qs, ks, vs = (t.unsqueeze(0).transpose(1, 2) for t in (q, k, v))
    o = torch.nn.functional.scaled_dot_product_attention(
        qs, ks, vs, is_causal=True, enable_gqa=True)
    return o.transpose(1, 2).squeeze(0)
