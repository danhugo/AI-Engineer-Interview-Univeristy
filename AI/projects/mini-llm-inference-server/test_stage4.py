"""Stage 4 gate: prefix caching must save work without changing the answer.

Run: ./sync.sh py test_stage4.py

Only *full* blocks are cached, so prompts must share at least block_size (256)
tokens before anything can hit. The test builds a 512-token shared prefix and
gives each prompt a different tail.

  1. correctness: logits with prefix caching on == with it off
  2. repeat:      running the same prompt twice hits the cache the second time
  3. sharing:     prompts with a common prefix hit it
  4. safety:      hashes are published only after K/V is written, so a batch
                  that shares a prefix must not read empty blocks
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_manager import build_manager_for
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from models.qwen3 import Qwen3ForCausalLM
from common import FILLER, path

TOL = 0.75      # bf16 noise; see NOTES.md
BLOCK = 256
PREFIX_LEN = 512  # exactly 2 blocks


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


def prefill(model, manager, prompts):
    """One prefill step. Returns (logits, cached-token count per sequence)."""
    scheduler = Scheduler(manager, max_num_seqs=64, max_num_batched_tokens=32768)
    engine = LLMEngine(model, manager, scheduler)
    for p in prompts:
        engine.add_request(p, 8, None)
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == len(prompts), f"took {len(seqs)}"
    hits = [s.num_cached for s in seqs]        # read before forward overwrites it
    logits = engine.forward_logits(seqs, True).float().cpu().clone()
    for s in seqs:
        manager.deallocate(s)
    return logits, hits


def main():
    tokenizer, model = load()

    base = tokenizer(FILLER, return_tensors="pt").input_ids[0].tolist()
    prefix = (base * (-(-PREFIX_LEN // len(base))))[:PREFIX_LEN]
    tails = [
        tokenizer(" So in summary, the key point is", return_tensors="pt").input_ids[0].tolist(),
        tokenizer(" Therefore the main tradeoff involves", return_tensors="pt").input_ids[0].tolist(),
        tokenizer(" Finally, one open question remains", return_tensors="pt").input_ids[0].tolist(),
    ]
    prompts = [prefix + t for t in tails]
    print(f"[prompts] shared prefix {PREFIX_LEN} tokens, total lengths "
          f"{[len(p) for p in prompts]}\n")

    # --- 1. reference: prefix caching disabled ---
    off = build_manager_for(model, num_blocks=64, block_size=BLOCK,
                            enable_prefix_caching=False)
    ref, ref_hits = prefill(model, off, prompts)
    print(f"[off] cached tokens per seq: {ref_hits}  hit rate {off.hit_rate:.1%}")
    assert all(h == 0 for h in ref_hits), "prefix caching was supposed to be off"

    # --- 2. prefix caching on, cold cache: nothing to hit yet ---
    on = build_manager_for(model, num_blocks=64, block_size=BLOCK,
                           enable_prefix_caching=True)
    cold, cold_hits = prefill(model, on, prompts)
    print(f"[cold] cached tokens per seq: {cold_hits}  hit rate {on.hit_rate:.1%}")
    # Hashes publish only after the forward, so same-batch siblings must miss.
    assert all(h == 0 for h in cold_hits), \
        "a sibling in the same batch hit an unwritten block — unsafe sharing"

    # --- 3. warm cache: the same prompts should now hit the shared prefix ---
    warm, warm_hits = prefill(model, on, prompts)
    print(f"[warm] cached tokens per seq: {warm_hits}  hit rate {on.hit_rate:.1%}")
    assert all(h >= PREFIX_LEN - BLOCK for h in warm_hits), \
        f"expected ~{PREFIX_LEN} cached tokens per sequence, got {warm_hits}"

    # --- 4. the answer must not change ---
    print()
    ok = True
    for i, (r, c, w) in enumerate(zip(ref, cold, warm)):
        d_cold = (r - c).abs().max().item()
        d_warm = (r - w).abs().max().item()
        top_ok = r.argmax().item() == w.argmax().item()
        bad = d_warm > TOL or not top_ok
        ok &= not bad
        print(f"  seq {i}: cold vs off {d_cold:6.4f}   warm vs off {d_warm:6.4f}"
              f"   top-1 {'same' if top_ok else 'DIFFERS'}   {'FAIL' if bad else 'ok'}")

    saved = sum(warm_hits)
    total = sum(len(p) for p in prompts)
    print(f"\n[savings] {saved} of {total} prompt tokens needed no attention "
          f"compute ({saved / total:.0%})")

    if ok:
        print("\n[stage4] PASS — prefix caching saves work, answer unchanged")
        return
    print("\n[stage4] FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
