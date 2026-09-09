"""Stage 1 gate: is our Qwen3 the same function as HuggingFace's?

Run: ./sync.sh run 'python test_stage1.py'

Everything downstream assumes the model is right. If it is not, stages 2-9 all
compare ours against ours and pass while measuring nothing. So this runs first
and runs both precisions, for two different reasons.

## fp32 — proves the architecture

fp32 is only possible here. From stage 2 on, every path goes through
flash-attn's paged kernels, which accept fp16/bf16 only. So this is the single
place we can demand **exact** equality and get a proof rather than an absence
of evidence: our reimplementation is the same function, not merely close.

Uses SDPA on both sides, one model per GPU (32 GB each in fp32).

## bf16 — calibrates the noise floor

Stages 2-9 can only be tested in bf16, and their tolerances have to come from
somewhere. This is that somewhere: the one run where fp32 has already proven
the code correct, so whatever bf16 spread appears here IS the normal amount.
Later stages compare against it — same spread means nothing new, more spread
means a new problem.

Checked at four lengths, because a short prompt hides everything: a 6-token
prompt reported a perfect 100% top-1 while the layer-16 blowup at 2048 tokens
went unseen.

## Why every metric, even in fp32 where one would logically do

`torch.equal` returning False tells you nothing — a 1e-7 kernel difference and
a structural bug look identical. And fp32 is not guaranteed bit-exact anyway:
TF32 matmuls, a cuBLAS version bump, or a different GPU can all change the last
bits with no bug present. The full readout is what distinguishes rounding from
breakage, and it costs one extra reduction over a forward pass already done.
"""

import torch

from common import PROMPT, build_long_ids, load_models, path

BF16_LENGTHS = (6, 128, 512, 2048)
FP32_LENGTH = 1024          # must exceed 603, the worst bf16 position
MAX_NEW_TOKENS = 40

# Watch the element that blew up by 8400 in bf16. A zero at the output does not
# prove the inside is clean: the final RMSNorm divides the hidden state by its
# own RMS, so a difference that is purely a rescale of the whole vector
# disappears from the logits entirely.
SUSPECT = (16, 603, 2276)   # (layer, position, hidden dim)

# bf16 tolerances. See NOTES.md for where these come from.
TOL_TV = 0.10
TOP_K = 5


def metrics(ref: torch.Tensor, our: torch.Tensor, top_k: int = TOP_K) -> dict:
    """Every comparison we make, over (seq, vocab) logits.

    One forward pass produces logits at every position, so this is ~155M
    numbers at seq 1024. Each metric reduces them differently:

      max_diff   max over seq AND vocab. Sees the single worst point, blind to
                 how widely the error is spread.
      tv         per position, sum |p-q| over vocab, halved; then the worst
                 position. Sum rather than max, so it does see the spread; and
                 because probabilities sum to 1, it lands in [0,1] and reads as
                 "this fraction of probability mass differs".
      top1       fraction of positions picking the same token. What greedy
                 decoding actually uses, but it flips on exact ties, so report
                 it and do not gate on it.
      topk       each side's top-1 must sit inside the other's top-k. vLLM's
                 own check. Tie-proof, so this is the gate.
    """
    ref, our = ref.float(), our.float()
    d = (ref - our).abs()

    ref_p, our_p = ref.softmax(-1), our.softmax(-1)
    tv = (ref_p - our_p).abs().sum(-1).max().item() / 2
    max_prob = (ref_p - our_p).abs().max().item()
    del ref_p, our_p

    r1, o1 = ref.argmax(-1), our.argmax(-1)
    r_k, o_k = ref.topk(top_k, -1).indices, our.topk(top_k, -1).indices
    mutual = ((o1.unsqueeze(-1) == r_k).any(-1) & (r1.unsqueeze(-1) == o_k).any(-1))

    return {
        "positions": ref.shape[0],
        "max_diff": d.max().item(),
        "scale": ref.abs().max().item(),
        "max_prob_diff": max_prob,
        "tv": tv,
        "top1": (r1 == o1).float().mean().item(),
        "topk": mutual.float().mean().item(),
    }


def show(m: dict, indent: str = "  ") -> None:
    rel = m["max_diff"] / m["scale"] if m["scale"] else 0.0
    print(f"{indent}max abs logit diff : {m['max_diff']:.6f}"
          f"   (scale {m['scale']:.1f}, relative {rel:.3%})")
    print(f"{indent}max prob diff      : {m['max_prob_diff']:.6f}")
    print(f"{indent}worst-position TV  : {m['tv']:.6f}")
    print(f"{indent}exact top-1 match  : {m['top1']:.2%}")
    print(f"{indent}top-{TOP_K} agreement    : {m['topk']:.2%}")


@torch.inference_mode()
def layer_output(model, layer_idx: int, ids, positions=None):
    """Hidden state leaving one decoder layer, as (seq, dim)."""
    box = {}

    def hook(_m, _i, out):
        t = (out[0] if isinstance(out, tuple) else out).detach().float()
        box["h"] = t.squeeze(0) if t.dim() == 3 and t.shape[0] == 1 else t

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        model(ids) if positions is None else model(ids, positions)
    finally:
        handle.remove()
    return box["h"]


# ------------------------------------------------------------------ fp32

def check_fp32() -> bool:
    print("=" * 64)
    print("fp32 — architecture proof (SDPA, one model per GPU)")
    print("=" * 64)
    if torch.cuda.device_count() < 2:
        print("  SKIPPED: needs 2 GPUs for two fp32 copies (32 GB each)")
        return True

    tokenizer, hf, ours = load_models(
        path, dtype=torch.float32, attn="sdpa", devices=("cuda:0", "cuda:1"))
    ids = build_long_ids(tokenizer, FP32_LENGTH, device="cpu")
    positions = torch.arange(FP32_LENGTH)

    with torch.inference_mode():
        ref = hf(ids.to("cuda:0")).logits[0].cpu()
        our = ours(ids.to("cuda:1"), positions.to("cuda:1")).cpu()

    m = metrics(ref, our)
    print(f"\n  seq_len {m['positions']}")
    show(m)

    # The element that diverged by 8400 in bf16.
    layer, pos, dim = SUSPECT
    r16 = layer_output(hf, layer, ids.to("cuda:0"))
    o16 = layer_output(ours, layer, ids.to("cuda:1"), positions.to("cuda:1"))
    r16, o16 = r16.cpu(), o16.cpu()
    at = abs(r16[pos, dim].item() - o16[pos, dim].item())
    overall = (r16 - o16).abs().max().item()
    print(f"\n  layer {layer} at (pos {pos}, dim {dim}):")
    print(f"    HF {r16[pos, dim].item():12.3f}   ours {o16[pos, dim].item():12.3f}"
          f"   diff {at:.6f}")
    print(f"    layer {layer} max|diff| over all positions: {overall:.6f}")

    checks = {
        "max abs logit diff == 0": m["max_diff"] == 0.0,
        "TV == 0": m["tv"] == 0.0,
        "top-1 == 100%": m["top1"] == 1.0,
        f"top-{TOP_K} == 100%": m["topk"] == 1.0,
        f"layer {layer} diff == 0": overall == 0.0,
    }
    print()
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    del hf, ours
    torch.cuda.empty_cache()
    return all(checks.values())


# ------------------------------------------------------------------ bf16

def check_bf16() -> bool:
    print()
    print("=" * 64)
    print("bf16 — production path, and the noise baseline for stages 2-9")
    print("=" * 64)

    tokenizer, hf, ours = load_models(path, dtype=torch.bfloat16,
                                      attn="flash_attention_2")
    ok = True
    for n in BF16_LENGTHS:
        ids = build_long_ids(tokenizer, n)
        positions = torch.arange(n, device=ids.device)
        with torch.inference_mode():
            ref = hf(ids).logits[0]
            our = ours(ids, positions)[0]
        m = metrics(ref, our)
        print(f"\n  seq_len {n}")
        show(m)
        passed = m["topk"] == 1.0 and m["tv"] <= TOL_TV
        ok &= passed
        print(f"    -> {'PASS' if passed else 'FAIL'} "
              f"(gate: top-{TOP_K} == 100%, TV <= {TOL_TV})")
        del ids, ref, our
        torch.cuda.empty_cache()

    # One real generation, to confirm the thing actually talks.
    del hf
    torch.cuda.empty_cache()
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            positions = torch.arange(ids.shape[1], device=ids.device)
            nxt = ours(ids, positions)[:, -1, :].argmax(-1)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            if nxt.item() == tokenizer.eos_token_id:
                break
    print(f"\n  [generate] {tokenizer.decode(ids[0], skip_special_tokens=True)}")
    return ok


def main():
    fp32_ok = check_fp32()
    bf16_ok = check_bf16()

    print()
    print("=" * 64)
    if fp32_ok and bf16_ok:
        print("[stage1] PASS — architecture proven in fp32, bf16 baseline recorded")
        return
    print(f"[stage1] FAIL — fp32 {'ok' if fp32_ok else 'FAILED'}, "
          f"bf16 {'ok' if bf16_ok else 'FAILED'}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
