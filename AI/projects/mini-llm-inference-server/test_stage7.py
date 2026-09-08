"""Stage 7 gate: our Triton flash attention must match the real one.

Run: ./sync.sh py test_stage7.py

Per the repo rule, from-scratch code is for study and gets a test proving it
matches the library. The engine never imports it.

Ground truth is fp32 SDPA, not flash-attn. Both kernels approximate the same
maths in bf16, so measuring one against the other cannot say which is closer.
Against an fp32 reference, ours should land in the same error band as
flash-attn — and because our kernel accumulates in fp32 throughout, it may be
slightly closer.
"""

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

from layers.flash_attn_study import flash_attn_study

# (batch, seq_q, seq_k, q_heads, kv_heads, head_dim, causal)
CASES = [
    ("qwen3 prefill",     2, 512, 512, 32, 8, 128, True),
    ("qwen3 long",        1, 2048, 2048, 32, 8, 128, True),
    ("decode-ish q=1",    4, 1, 777, 32, 8, 128, True),
    ("non-causal",        2, 256, 256, 16, 16, 64, False),
    ("MHA (no GQA)",      2, 320, 320, 8, 8, 64, True),
    ("ragged seq len",    1, 300, 300, 4, 2, 128, True),
    ("head_dim 32",       2, 128, 128, 8, 2, 32, True),
]


def reference(q, k, v, causal):
    """fp32 SDPA. Slow and memory-hungry, but the accurate answer.

    Do NOT use is_causal=True here. When seq_q != seq_k, SDPA aligns the mask
    top-left, so a single decode query would see only token 0. flash-attn (and
    our kernel) align bottom-right, so that query sees the whole history. The
    first version of this test used is_causal and reported an error of 3.57 for
    BOTH kernels — two independent kernels cannot be wrong identically, which
    is what gave the reference away. Build the mask explicitly instead.
    """
    m, n = q.shape[1], k.shape[1]
    qs, ks, vs = (t.float().transpose(1, 2) for t in (q, k, v))
    mask = None
    if causal:
        shift = n - m
        i = torch.arange(m, device=q.device)[:, None]
        j = torch.arange(n, device=q.device)[None, :]
        mask = (i + shift) >= j
    o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, enable_gqa=True)
    return o.transpose(1, 2)


def main():
    torch.manual_seed(0)
    print(f"{'case':18} {'shape':>22}  {'ours':>9}  {'flash':>9}  {'verdict':>8}")

    ok = True
    for name, b, m, n, h, hkv, d, causal in CASES:
        q = torch.randn(b, m, h, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, n, hkv, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(b, n, hkv, d, device="cuda", dtype=torch.bfloat16)

        ref = reference(q, k, v, causal)
        ours = flash_attn_study(q, k, v, causal=causal)

        # flash-attn's own causal masking is bottom-right aligned, same as ours.
        theirs = flash_attn_func(q, k, v, causal=causal)

        e_ours = (ours.float() - ref).abs().max().item()
        e_flash = (theirs.float() - ref).abs().max().item()

        # Ours must be in the same band as flash-attn, allowing headroom.
        bad = not (e_ours <= max(4 * e_flash, 0.02))
        ok &= not bad
        shape = f"b{b} q{m} k{n} h{h}/{hkv} d{d}"
        print(f"{name:18} {shape:>22}  {e_ours:9.5f}  {e_flash:9.5f}  "
              f"{'FAIL' if bad else 'ok':>8}")

    # A quick look at whether it is in the right performance ballpark. A study
    # kernel with no autotuning will lose; the point is that it is not 100x off.
    import time
    q = torch.randn(4, 2048, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(4, 2048, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(4, 2048, 8, 128, device="cuda", dtype=torch.bfloat16)

    def bench(fn, iters=20):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    t_ours = bench(lambda: flash_attn_study(q, k, v, causal=True))
    t_flash = bench(lambda: flash_attn_func(q, k, v, causal=True))
    print(f"\n[speed] b4 seq2048 h32/8 d128 causal: "
          f"ours {t_ours:.2f} ms, flash-attn {t_flash:.2f} ms "
          f"({t_ours / t_flash:.1f}x slower)")

    print()
    if ok:
        print("[stage7] PASS — our Triton kernel matches flash-attn's accuracy")
        return
    print("[stage7] FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
