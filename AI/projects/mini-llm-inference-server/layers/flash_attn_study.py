"""Flash attention, written from scratch in Triton. STUDY ONLY.

The engine imports `layers.attention`, never this file. Its job is to make the
algorithm concrete, with `test_stage7.py` proving it matches the real
`flash-attn` kernel.

## The problem

Plain attention materialises the score matrix S = QK^T, which is
(seq x seq). At 8k context that is 64M numbers per head — it does not fit in
SRAM, so every element is written to HBM and read back for the softmax and
again for the PV product. Attention becomes memory-bound, and memory traffic
grows with seq^2.

## The idea

Never materialise S. Walk K/V in tiles that *do* fit in SRAM, and keep a
running softmax. The obstacle is that softmax needs a global max and a global
sum, which you do not know until you have seen every tile.

Online softmax fixes that. Track the running max `m` and running sum `l`. When
a tile pushes the max higher, rescale what you already have by
`exp(m_old - m_new)` and carry on:

    m_new = max(m_old, max(tile))
    alpha = exp(m_old - m_new)          # correction for everything so far
    l     = l * alpha + sum(exp(tile - m_new))
    acc   = acc * alpha + exp(tile - m_new) @ V_tile

At the end `acc / l` is exactly the softmax attention output. Subtracting the
max keeps `exp` from overflowing, which is why the max is tracked at all.

Traffic drops from O(seq^2) to O(seq * head_dim): each Q tile is read once,
each K/V tile once per Q tile, and S never leaves SRAM. Same arithmetic, same
answer — it is a memory-access rewrite, not an approximation.

FlashAttention-2 detail: rescale the *accumulator* and divide by `l` once at
the end, rather than renormalising every tile. Fewer non-matmul operations,
which is what limits throughput on a tensor-core GPU.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd(
    Q, K, V, O,
    # strides, in elements: batch, seq, head, dim
    sqb, sqm, sqh, sqd,
    skb, skn, skh, skd,
    svb, svn, svh, svd,
    sob, som, soh, sod,
    M, N, scale,
    H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
):
    """One program handles one Q tile of one (batch, head)."""
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    # Grouped-query attention: several q heads share one kv head.
    hkv = h // (H // HKV)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Load this Q tile once and keep it in SRAM for the whole K/V walk.
    q = tl.load(
        Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=offs_m[:, None] < M, other=0.0,
    ).to(tl.float32)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), tl.float32)                # running sum
    acc = tl.zeros((BLOCK_M, D), tl.float32)              # running output

    # When seq_q < seq_k the queries align to the END of the keys, which is
    # flash-attn's bottom-right convention: a single decode query attends to
    # the whole history. Row i therefore sees keys up to i + (N - M).
    shift = N - M
    # Causal masking means a Q tile never needs K past its own last row, so
    # the loop stops early instead of masking work it already did.
    hi = tl.minimum(N, (start_m + 1) * BLOCK_M + shift) if CAUSAL else N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k = tl.load(
            K + b * skb + hkv * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=offs_n[:, None] < N, other=0.0,
        ).to(tl.float32)

        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(offs_n[None, :] < N, s, float("-inf"))   # past the end
        if CAUSAL:
            s = tl.where(offs_m[:, None] + shift >= offs_n[None, :], s, float("-inf"))

        # --- online softmax update ---
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)          # rescales everything so far
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            V + b * svb + hkv * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
            mask=offs_n[:, None] < N, other=0.0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new

    # Normalise once, at the end. This is the FlashAttention-2 ordering.
    acc = acc / l_i[:, None]
    tl.store(
        O + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
        acc.to(O.dtype.element_ty), mask=offs_m[:, None] < M,
    )


def flash_attn_study(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     causal: bool = True, softmax_scale: float | None = None,
                     block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """Causal GQA attention. Same signature and layout as flash_attn_func.

    q: (batch, seq_q, heads,    head_dim)
    k: (batch, seq_k, kv_heads, head_dim)
    v: (batch, seq_k, kv_heads, head_dim)
    """
    assert q.dim() == k.dim() == v.dim() == 4
    b, m, h, d = q.shape
    n, hkv = k.shape[1], k.shape[2]
    assert k.shape == v.shape, "k and v must match"
    assert q.shape[0] == k.shape[0] and q.shape[3] == k.shape[3]
    assert h % hkv == 0, f"{h} q heads not a multiple of {hkv} kv heads"
    assert m <= n, "seq_q must not exceed seq_k (causal aligns q to the end)"
    assert d in (16, 32, 64, 128, 256), f"head_dim {d} must be a power of two"

    o = torch.empty_like(q)
    scale = softmax_scale if softmax_scale is not None else d ** -0.5

    grid = (triton.cdiv(m, block_m), b * h)
    _flash_fwd[grid](
        q, k, v, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        m, n, scale,
        H=h, HKV=hkv, D=d,
        BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
    )
    return o
