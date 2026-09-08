"""Stage 3 gate: continuous batching must not change the computation.

Run: ./sync.sh py test_stage3.py

What NOT to test: that batched generation produces the same *text* as
unbatched. It does not, and that is fine. Greedy decoding hits exact bf16 ties
— `diag_batch.py` found two tokens both at logit 24.2500 — and batch size
changes reduction order inside the attention kernels, which flips the tie into
completely different text. Comparing trajectories tests chaos, not correctness.

What we test instead: given the *same* token history, batching must produce the
same logits. That isolates the machinery (cu_seqlens, slot mapping, block
tables, padding) from autoregressive amplification.

  1. prefill: 6 uneven prompts in one batch vs one at a time
  2. decode:  same, one step, histories forced identical
  3. blocks do not leak; preemption round-trips and still generates
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_manager import build_manager_for
from engine.generate import generate_many
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from models.qwen3 import Qwen3ForCausalLM
from run import path

MAX_NEW = 24
# bf16 noise for this model; see NOTES.md. Logits run to ~55.
TOL = 0.75

# Uneven lengths on purpose: equal lengths would hide cu_seqlens bugs.
PROMPTS = [
    "The specialty of Hanoi is",
    "In one sentence, explain why paged attention saves memory:",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time in a city of lakes,",
    "def fibonacci(n):",
    "The three most important ideas in an LLM inference server are",
]


def load():
    tokenizer = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model = Qwen3ForCausalLM(hf.config).to(torch.bfloat16)
    model.load_state_dict(hf.state_dict())
    model = model.cuda().eval()
    del hf
    torch.cuda.empty_cache()
    return tokenizer, model


def run_steps(model, manager, prompts, forced_first=None):
    """Prefill these prompts, then take one decode step.

    Returns (prefill_logits, decode_logits), both (num_seqs, vocab).

    forced_first pins the token appended after prefill. Without it, an exact
    bf16 tie can make the batched and unbatched runs decode from different
    histories, and then the decode comparison is meaningless.
    """
    scheduler = Scheduler(manager, max_num_seqs=64, max_num_batched_tokens=16384)
    engine = LLMEngine(model, manager, scheduler)
    for prompt in prompts:
        engine.add_request(prompt, MAX_NEW, None)

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == len(prompts), \
        f"scheduler took {len(seqs)} of {len(prompts)}"
    pre = engine.forward_logits(seqs, is_prefill=True)

    # The cache now holds each prompt's K/V, so a decode step is meaningful.
    tokens = forced_first if forced_first is not None else pre.argmax(-1).tolist()
    for seq, token in zip(seqs, tokens):
        seq.append(token)
        manager.allocate(seq)  # may need one more block
    dec = engine.forward_logits(seqs, is_prefill=False)

    out = pre.float().cpu().clone(), dec.float().cpu().clone()
    for seq in seqs:
        manager.deallocate(seq)
    return out


def compare(tokenizer, batched, alone, label):
    """Per-sequence logit agreement, tolerant of exact ties."""
    ok = True
    for i, (b, a) in enumerate(zip(batched, alone)):
        diff = (b - a).abs().max().item()
        same_top1 = b.argmax().item() == a.argmax().item()
        # An exact tie at the top means argmax is arbitrary, not wrong.
        top2 = a.topk(2).values
        tied = (top2[0] - top2[1]).item() < TOL
        status = "ok" if diff <= TOL else "FAIL"
        note = ""
        if not same_top1:
            note = "  (top-1 differs, but top-2 is a tie)" if tied else "  (top-1 DIFFERS)"
            if not tied:
                status = "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{label}] seq {i}: max|diff| {diff:7.4f}  {status}{note}")
    return ok


def main():
    tokenizer, model = load()
    eos = tokenizer.eos_token_id
    prompt_ids = [tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in PROMPTS]
    print(f"[prompts] lengths {[len(p) for p in prompt_ids]}\n")

    manager = build_manager_for(model, num_blocks=64, block_size=256)

    # --- all six in one batch ---
    pre_b, dec_b = run_steps(model, manager, prompt_ids)
    forced = pre_b.argmax(-1).tolist()  # pin histories for the decode compare
    assert manager.num_free_blocks == manager.num_blocks, "blocks leaked (batched)"

    # --- each one alone, decoding from the same forced token ---
    singles = [run_steps(model, manager, [p], forced_first=[t])
               for p, t in zip(prompt_ids, forced)]
    pre_a = torch.cat([a for a, _ in singles])
    dec_a = torch.cat([d for _, d in singles])
    assert manager.num_free_blocks == manager.num_blocks, "blocks leaked (alone)"

    ok_prefill = compare(tokenizer, pre_b, pre_a, "prefill")
    ok_decode = compare(tokenizer, dec_b, dec_a, "decode ")

    # --- 3. end-to-end smoke, plus preemption on a 3-block pool ---
    texts = generate_many(model, manager, prompt_ids, MAX_NEW, eos)
    assert manager.num_free_blocks == manager.num_blocks, "blocks leaked (generate)"

    tiny = build_manager_for(model, num_blocks=3, block_size=256)
    pre = generate_many(model, tiny, prompt_ids, MAX_NEW, eos, max_num_seqs=8)
    assert tiny.num_free_blocks == tiny.num_blocks, "blocks leaked (preempt)"
    ok_preempt = all(len(t) > len(p) for t, p in zip(pre, prompt_ids))
    print(f"\n  [preempt] 6 prompts through a 3-block pool: "
          f"{'all produced tokens' if ok_preempt else 'SOME PRODUCED NOTHING'}")

    print()
    for p, t in zip(PROMPTS, texts):
        new = tokenizer.decode(t[len(tokenizer(p).input_ids):], skip_special_tokens=True)
        print(f"  {p!r}\n    -> {new.strip()!r}")

    if ok_prefill and ok_decode and ok_preempt:
        print("\n[stage3] PASS — batching leaves the computation unchanged")
        return
    print("\n[stage3] FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
