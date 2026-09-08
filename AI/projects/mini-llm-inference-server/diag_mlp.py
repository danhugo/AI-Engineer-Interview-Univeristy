"""Is the layer-16 MLP gap a bug in our code, or bf16 cancellation?

Run: ./sync.sh py diag_mlp.py

Feeds HF's own layer-16 MLP input to BOTH MLPs, so all upstream accumulation is
removed and only the MLP implementation is under test.

  bf16 differs, fp32 agrees  -> same math; bf16 rounding amplified by
                                cancellation in down_proj. Not a bug.
  fp32 also differs          -> a real bug in our MLP.
"""

import torch

from run import build_long_ids, load_models, path

SEQ_LEN = 2048
LAYER = 16
POS, DIM = 603, 2276


def main():
    tokenizer, hf, ours = load_models(path)
    ids = build_long_ids(tokenizer, SEQ_LEN)
    positions = torch.arange(ids.shape[1], device=ids.device)

    hf_mlp = hf.model.layers[LAYER].mlp
    our_mlp = ours.model.layers[LAYER].mlp

    # 0. Are the weights even the same? If not, nothing else matters.
    print("=== weight check ===")
    for name in ("gate_proj", "up_proj", "down_proj"):
        a = getattr(hf_mlp, name).weight
        b = getattr(our_mlp, name).weight
        same = torch.equal(a, b)
        print(f"  {name:11s} identical: {same}   shape {tuple(a.shape)}")
        assert same, f"{name} weights differ — load_state_dict problem, not precision"

    # 1. Capture HF's real input to that MLP.
    box = {}
    h = hf.model.layers[LAYER].post_attention_layernorm.register_forward_hook(
        lambda _m, _i, out: box.__setitem__("x", out.detach())
    )
    with torch.inference_mode():
        hf(ids)
    h.remove()
    x = box["x"]
    print(f"\nMLP input: shape {tuple(x.shape)}, dtype {x.dtype}, |max| {x.abs().max().item():.2f}")

    # 2. Same input through both MLPs, in bf16.
    with torch.inference_mode():
        a = hf_mlp(x).float()
        b = our_mlp(x).float()
    print(f"\n=== bf16, identical input ===")
    print(f"  max|diff|      {(a - b).abs().max().item():12.4f}")
    print(f"  at ({POS},{DIM})  HF {a[0, POS, DIM].item():12.3f}   ours {b[0, POS, DIM].item():12.3f}")

    # 3. Same input through both MLPs, in fp32.
    with torch.inference_mode():
        a32 = hf_mlp.float()(x.float()).float()
        b32 = our_mlp.float()(x.float()).float()
    print(f"\n=== fp32, identical input ===")
    print(f"  max|diff|      {(a32 - b32).abs().max().item():12.6f}")
    print(f"  at ({POS},{DIM})  HF {a32[0, POS, DIM].item():12.3f}   ours {b32[0, POS, DIM].item():12.3f}")
    print(f"\n  fp32 truth vs bf16:  HF {a[0, POS, DIM].item():10.2f}"
          f"   ours {b[0, POS, DIM].item():10.2f}"
          f"   fp32 {a32[0, POS, DIM].item():10.2f}")

    # 4. Quantify the cancellation at that one output element.
    #    down_proj row . mlp_hidden  ==  sum of 12288 products.
    with torch.inference_mode():
        xf = x.float()
        hidden = torch.nn.functional.silu(hf_mlp.gate_proj(xf)) * hf_mlp.up_proj(xf)
        terms = hf_mlp.down_proj.weight[DIM].float() * hidden[0, POS]
        total, absum = terms.sum().item(), terms.abs().sum().item()
    print(f"\n=== cancellation at ({POS},{DIM}) ===")
    print(f"  sum of terms      {total:14.2f}")
    print(f"  sum of |terms|    {absum:14.2f}")
    print(f"  amplification     {absum / abs(total):14.1f}x")
    print(f"  bf16 precision ~0.4%, so expected relative error here:"
          f" {0.004 * absum / abs(total):.1%}")


if __name__ == "__main__":
    main()
