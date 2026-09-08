"""Why does prompt 0 differ between batch sizes?

Run: ./sync.sh py diag_batch.py

test_stage3 compared batched output against "each prompt alone" and found one
divergence. But the batched answer matches stage 1 and stage 2, so "alone" was
the wrong reference. This compares all three against the uncached path, which
is the only trusted ground truth, and then asks the real question:

  is the divergence token a near-tie?

If the top-2 logits at that step are within bf16 noise (~0.5 for this model),
then greedy simply picked the other side of a coin flip and no code is wrong.
Batch size changes reduction order inside the attention kernels, which is
enough to flip a tie.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_manager import build_manager_for
from engine.generate import generate_many
from models.qwen3 import Qwen3ForCausalLM
from run import path
from test_stage3 import MAX_NEW, PROMPTS


@torch.inference_mode()
def uncached(model, ids, max_new_tokens, eos):
    """Trusted reference: recompute everything, no cache, batch of one."""
    for _ in range(max_new_tokens):
        positions = torch.arange(ids.shape[1], device=ids.device)
        nxt = model(ids, positions)[:, -1, :].argmax(-1)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        if nxt.item() == eos:
            break
    return ids[0].tolist()


@torch.inference_mode()
def logits_at(model, prefix_ids, device):
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(prefix_ids), device=device)
    return model(ids, positions)[0, -1].float()


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def main():
    tokenizer = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model = Qwen3ForCausalLM(hf.config).to(torch.bfloat16)
    model.load_state_dict(hf.state_dict())
    model = model.cuda().eval()
    del hf
    torch.cuda.empty_cache()

    eos = tokenizer.eos_token_id
    prompt_ids = [tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in PROMPTS]
    p0 = prompt_ids[0]

    ref = uncached(model, torch.tensor([p0], device="cuda"), MAX_NEW, eos)

    manager = build_manager_for(model, num_blocks=64, block_size=256)
    b1 = generate_many(model, manager, [p0], MAX_NEW, eos)[0]
    b6 = generate_many(model, manager, prompt_ids, MAX_NEW, eos)[0]

    print("=== prompt 0, three ways ===")
    for label, toks in (("uncached", ref), ("engine batch=1", b1), ("engine batch=6", b6)):
        print(f"  {label:15s} {tokenizer.decode(toks, skip_special_tokens=True)!r}")

    print("\n=== agreement with the uncached reference ===")
    for label, toks in (("batch=1", b1), ("batch=6", b6)):
        d = first_diff(ref, toks)
        print(f"  {label}: {'identical' if d is None else f'diverges at token {d}'}")

    # Where any pair first disagrees, inspect the decision.
    cand = [d for d in (first_diff(ref, b1), first_diff(ref, b6), first_diff(b1, b6))
            if d is not None]
    if not cand:
        print("\nall three identical — nothing to explain")
        return
    pos = min(cand)

    prefix = ref[:pos]
    lg = logits_at(model, prefix, "cuda")
    top = lg.topk(5)
    gap = (top.values[0] - top.values[1]).item()

    print(f"\n=== the decision at token {pos} ===")
    print(f"  context: {tokenizer.decode(prefix[-12:], skip_special_tokens=True)!r}")
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        picks = [n for n, t in (("uncached", ref), ("b1", b1), ("b6", b6))
                 if pos < len(t) and t[pos] == i]
        mark = ("  <- " + ", ".join(picks)) if picks else ""
        print(f"    {v:9.4f}  {tokenizer.decode([i])!r}{mark}")
    print(f"\n  top-1 minus top-2 gap: {gap:.4f}")
    print("  bf16 noise for this model is ~0.3-0.6 (see NOTES.md)")
    print(f"  verdict: {'NEAR-TIE, not a bug' if gap < 0.6 else 'gap too large - real bug'}")


if __name__ == "__main__":
    main()
