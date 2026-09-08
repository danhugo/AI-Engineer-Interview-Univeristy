"""Definitive check: do we match HuggingFace in fp32?

Run: ./sync.sh py diag_fp32.py

The bf16 run diverges hugely at two 'massive activation' positions. The layer-16
MLP was proven bit-identical given the same input, so the suspicion is that bf16
precision alone explains it. This tests that directly.

  fp32 agrees closely -> our implementation is correct; bf16 is the whole story.
  fp32 also diverges  -> a real bug that bf16 was masking.

One model per GPU (8B fp32 is ~32GB each, the box has 2x80GB). fp32 forces the
SDPA path since flash-attn only takes fp16/bf16.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.qwen3 import Qwen3ForCausalLM
from run import FILLER, path

SEQ_LEN = 1024  # must exceed 603, the worst bf16 position
POS, DIM = 603, 2276


def main():
    assert torch.cuda.device_count() >= 2, "needs 2 GPUs for two fp32 copies"
    tokenizer = AutoTokenizer.from_pretrained(path)

    # sdpa on both sides: fp32 rules out flash-attn entirely.
    hf = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda:0").eval()

    ours = Qwen3ForCausalLM(hf.config).to(torch.float32)
    ours.load_state_dict(hf.state_dict())
    ours = ours.to("cuda:1").eval()

    base = tokenizer(FILLER, return_tensors="pt").input_ids[0]
    ids = base.repeat(-(-SEQ_LEN // len(base)))[:SEQ_LEN].unsqueeze(0)
    positions = torch.arange(SEQ_LEN)

    with torch.inference_mode():
        ref = hf(ids.to("cuda:0")).logits[0].float().cpu()
        our = ours(ids.to("cuda:1"), positions.to("cuda:1"))[0].float().cpu()

    d = (ref - our).abs()
    scale = ref.abs().max().item()
    ref_p, our_p = ref.softmax(-1), our.softmax(-1)

    print(f"=== fp32, seq_len {SEQ_LEN} ===")
    print(f"  max abs logit diff : {d.max().item():.6f}   (logit scale {scale:.1f},"
          f" relative {d.max().item() / scale:.3%})")
    print(f"  max prob diff      : {(ref_p - our_p).abs().max().item():.6f}")
    print(f"  worst-position TV  : {(ref_p - our_p).abs().sum(-1).max().item() / 2:.6f}")
    print(f"  exact top-1 match  : {(ref.argmax(-1) == our.argmax(-1)).float().mean().item():.2%}")
    print(f"  diff at position {POS}: {d[POS].max().item():.6f}"
          f"   (median over positions: {d.amax(-1).median().item():.6f})")

    # For contrast, this is what the same position looked like in bf16: ~8400.
    print(f"\n  hidden-state check at layer 16, position {POS}, dim {DIM}:")
    box = {}
    h = ours.model.layers[16].register_forward_hook(
        lambda _m, _i, out: box.__setitem__("o", (out[0] if isinstance(out, tuple) else out).detach())
    )
    h2 = hf.model.layers[16].register_forward_hook(
        lambda _m, _i, out: box.__setitem__("r", (out[0] if isinstance(out, tuple) else out).detach())
    )
    with torch.inference_mode():
        hf(ids.to("cuda:0"))
        ours(ids.to("cuda:1"), positions.to("cuda:1"))
    h.remove(); h2.remove()
    r16 = box["r"].float().cpu().squeeze(0)
    o16 = box["o"].float().cpu().squeeze(0)
    print(f"    HF {r16[POS, DIM].item():12.3f}   ours {o16[POS, DIM].item():12.3f}"
          f"   diff {abs(r16[POS, DIM].item() - o16[POS, DIM].item()):.4f}")
    print(f"    layer-16 max|diff| over all positions: {(r16 - o16).abs().max().item():.4f}")


if __name__ == "__main__":
    main()
