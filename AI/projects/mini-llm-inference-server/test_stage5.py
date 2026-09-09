"""Stage 5 gate: tensor parallelism must not change the answer.

Run:  ./sync.sh run 'python test_stage5.py'                    # save TP=1 reference
      ./sync.sh run 'torchrun --nproc_per_node=2 test_stage5.py'  # compare TP=2

Or in one go:  ./sync.sh run 'bash run_stage5.sh'

Sharding is a pure rearrangement: q/k/v and gate/up split their output
dimension, o_proj and down_proj split their input and all-reduce the partial
sums. The maths is identical, so TP=2 logits must match TP=1 within bf16 noise.

Each rank also owns only its own KV heads, so the cache is split too — 2 ranks
means each holds half the K/V. That is the real win: cache capacity scales with
the number of GPUs.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_manager import build_manager_for
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from models.qwen3 import Qwen3ForCausalLM
from utils import parallel
from utils.loader import load_weights
from common import path

REF_FILE = "/tmp/stage5_tp1_logits.pt"
TOL = 0.75  # bf16 noise; see NOTES.md

PROMPTS = [
    "The specialty of Hanoi is",
    "In one sentence, explain why paged attention saves memory:",
    "Q: What is 17 * 23?\nA:",
    "def fibonacci(n):",
]


def build(rank, world):
    tokenizer = AutoTokenizer.from_pretrained(path)
    # Load the checkpoint on CPU: every rank needs the full tensors to slice
    # from, and two 16GB copies on one GPU would be wasteful.
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    config = hf.config

    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    load_weights(model, hf.state_dict())
    del hf
    model = model.cuda().eval()
    torch.cuda.empty_cache()
    return tokenizer, model


def logits_for(model, manager, prompt_ids):
    scheduler = Scheduler(manager, max_num_seqs=64, max_num_batched_tokens=16384)
    engine = LLMEngine(model, manager, scheduler)
    for p in prompt_ids:
        engine.add_request(p, 8, None)
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == len(prompt_ids)
    pre = engine.forward_logits(seqs, True)

    for seq, tok in zip(seqs, pre.argmax(-1).tolist()):
        seq.append(tok)
        manager.allocate(seq)
    dec = engine.forward_logits(seqs, False)

    out = pre.float().cpu().clone(), dec.float().cpu().clone()
    for seq in seqs:
        manager.deallocate(seq)
    return out


def main():
    rank, world = parallel.init()
    tokenizer, model = build(rank, world)

    prompt_ids = [tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in PROMPTS]
    manager = build_manager_for(model, num_blocks=64, block_size=256)

    attn = model.model.layers[0].self_attn
    parallel.log(f"[tp] world={world}  heads/rank={attn.num_heads}  "
                 f"kv_heads/rank={attn.num_kv_heads}")
    parallel.log(f"[tp] KV cache {manager.bytes_per_token() / 1024:.0f} KB/token "
                 f"per rank ({manager.capacity_tokens()} tokens)")
    params = sum(p.numel() for p in model.parameters())
    parallel.log(f"[tp] {params / 1e9:.2f}B parameters held per rank")

    pre, dec = logits_for(model, manager, prompt_ids)

    if world == 1:
        torch.save({"pre": pre, "dec": dec}, REF_FILE)
        print(f"[tp1] saved reference to {REF_FILE}")
        print(f"[tp1] first tokens: {pre.argmax(-1).tolist()}")
        return

    # --- TP=2: compare against the saved single-GPU reference ---
    if not parallel.is_main():
        parallel.shutdown()
        return

    assert os.path.exists(REF_FILE), f"run TP=1 first to create {REF_FILE}"
    ref = torch.load(REF_FILE)

    ok = True
    print()
    for label, mine, theirs in (("prefill", pre, ref["pre"]), ("decode ", dec, ref["dec"])):
        for i in range(mine.shape[0]):
            diff = (mine[i] - theirs[i]).abs().max().item()
            top_ok = mine[i].argmax().item() == theirs[i].argmax().item()
            top2 = theirs[i].topk(2).values
            tied = (top2[0] - top2[1]).item() < TOL
            bad = diff > TOL or (not top_ok and not tied)
            ok &= not bad
            note = "" if top_ok else ("  (top-1 differs, top-2 tied)" if tied
                                      else "  (top-1 DIFFERS)")
            print(f"  [{label}] seq {i}: max|diff| {diff:7.4f}  "
                  f"{'FAIL' if bad else 'ok'}{note}")

    print(f"\n[tp2] first tokens: {pre.argmax(-1).tolist()}")
    print(f"[tp1] first tokens: {ref['pre'].argmax(-1).tolist()}")

    if ok:
        print("\n[stage5] PASS — TP=2 matches TP=1")
    else:
        print("\n[stage5] FAIL")
    parallel.shutdown()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
