"""Stage 6 gate: CUDA graphs must be faster and identical.

Run: ./sync.sh py test_stage6.py

A replayed graph runs exactly the kernels that were recorded, so decode logits
must match the eager path — and unlike everything else in this project the
match should be **bit-exact**, not just within bf16 noise: same kernels, same
order, same addresses.

Then measure. Decode is bound by kernel-launch overhead, so the win should be
large at small batch sizes (little GPU work to hide the launches behind) and
shrink as the batch grows.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_manager import build_manager_for
from engine.cuda_graph import DecodeGraphRunner
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from models.qwen3 import Qwen3ForCausalLM
from common import path

PROMPT = "The specialty of Hanoi is"
BATCHES = (1, 4, 16)
ITERS = 30


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


def prime(engine, scheduler, manager, prompt_ids, batch):
    """Admit `batch` sequences and prefill them, leaving them mid-decode."""
    for _ in range(batch):
        engine.add_request(list(prompt_ids), 10_000, None)
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == batch
    logits = engine.forward_logits(seqs, True)
    for seq, tok in zip(seqs, logits.argmax(-1).tolist()):
        seq.append(tok)
        manager.allocate(seq)
    return seqs


@torch.inference_mode()
def bench(engine, seqs, iters):
    """Time `iters` decode steps, without advancing the sequences."""
    engine.forward_logits(seqs, False)  # warm up
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        engine.forward_logits(seqs, False)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms/step


def main():
    tokenizer, model = load()
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0].tolist()

    manager = build_manager_for(model, num_blocks=64, block_size=256)

    print("[capture] recording decode graphs for batch sizes 1,2,4,8,16,32 ...")
    t0 = time.perf_counter()
    runner = DecodeGraphRunner(model, manager, max_batch=32, max_blocks_per_seq=4)
    runner.capture()
    print(f"[capture] done in {time.perf_counter() - t0:.1f}s\n")

    print(f"{'batch':>6}  {'eager ms':>9}  {'graph ms':>9}  {'speedup':>8}"
          f"  {'max|diff|':>10}  {'tok/s eager':>11}  {'tok/s graph':>11}")
    ok = True
    for batch in BATCHES:
        # Two engines over the same cache: one eager, one graph-backed.
        sched_e = Scheduler(manager, max_num_seqs=64, max_num_batched_tokens=16384)
        eager = LLMEngine(model, manager, sched_e)
        seqs = prime(eager, sched_e, manager, prompt_ids, batch)

        with torch.inference_mode():
            want = eager.forward_logits(seqs, False).float().clone()

        sched_g = Scheduler(manager, max_num_seqs=64, max_num_batched_tokens=16384)
        graphed = LLMEngine(model, manager, sched_g, graph_runner=runner)
        graphed.scheduler.running.extend(seqs)
        with torch.inference_mode():
            got = graphed.forward_logits(seqs, False).float().clone()

        diff = (want - got).abs().max().item()
        if diff != 0.0:
            ok = False

        ms_e = bench(eager, seqs, ITERS)
        ms_g = bench(graphed, seqs, ITERS)
        print(f"{batch:6d}  {ms_e:9.2f}  {ms_g:9.2f}  {ms_e / ms_g:7.2f}x"
              f"  {diff:10.6f}  {batch / ms_e * 1000:11.0f}  {batch / ms_g * 1000:11.0f}")

        for seq in seqs:
            manager.deallocate(seq)

    print()
    if ok:
        print("[stage6] PASS — graph replay is bit-exact and faster")
        return
    print("[stage6] FAIL — graph output differs from eager")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
