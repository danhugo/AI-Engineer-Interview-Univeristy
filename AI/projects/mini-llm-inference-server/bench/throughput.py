"""Stage 9: how fast is it, and where does the speed come from?

Run: ./sync.sh run 'python -m bench.throughput'
Then: ./sync.sh pull      # brings bench/results/ back

Four measurements, each isolating one thing:

  1. prefill  tokens/s vs prompt length — should be compute-bound, so roughly
              flat per token once the GPU is saturated
  2. decode   tokens/s vs batch size, graphs on and off — launch-bound, so the
              per-step time barely moves as the batch grows
  3. vs HF    the same end-to-end workload through transformers' generate()
  4. prefix   cache warm vs cold on a shared prompt prefix

Everything runs greedy so nothing depends on sampling luck.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_manager import build_manager_for
from engine.cuda_graph import DecodeGraphRunner
from engine.generate import generate_many
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from models.qwen3 import Qwen3ForCausalLM

MODEL = "Qwen/Qwen3-8B"
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")

PREFILL_LENS = (128, 512, 2048)
DECODE_BATCHES = (1, 4, 16, 32)
DECODE_STEPS = 30
E2E_BATCHES = (1, 8)
E2E_NEW_TOKENS = 64


def timed(fn, iters):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def load_ours():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = Qwen3ForCausalLM(hf.config).to(torch.bfloat16)
    model.load_state_dict(hf.state_dict())
    model = model.cuda().eval()
    del hf
    torch.cuda.empty_cache()
    return tokenizer, model


def prime(engine, scheduler, manager, prompt_ids, batch):
    """Admit `batch` sequences and prefill them, leaving them mid-decode."""
    for _ in range(batch):
        engine.add_request(list(prompt_ids), 10 ** 6, None)
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == batch, f"scheduled {len(seqs)}/{batch}"
    logits = engine.forward_logits(seqs, True)
    for seq, tok in zip(seqs, logits.argmax(-1).tolist()):
        seq.append(tok)
        manager.allocate(seq)
    return seqs


def bench_prefill(model, tokenizer, manager, filler, results):
    print("\n=== 1. prefill ===")
    print(f"{'prompt tokens':>14}  {'ms':>8}  {'tokens/s':>10}")
    for n in PREFILL_LENS:
        ids = (filler * (-(-n // len(filler))))[:n]
        rows = []

        def once():
            sched = Scheduler(manager, max_num_seqs=8, max_num_batched_tokens=65536)
            eng = LLMEngine(model, manager, sched)
            eng.add_request(list(ids), 4, None)
            seqs, _ = sched.schedule()
            eng.forward_logits(seqs, True)
            rows.append(seqs)

        # Fresh blocks each iteration, and prefix caching off for a clean number.
        manager.enable_prefix_caching = False
        sec = timed(lambda: (once(), [manager.deallocate(s) for s in rows.pop()]), 5)
        manager.enable_prefix_caching = True
        tps = n / sec
        print(f"{n:>14}  {sec * 1000:8.1f}  {tps:10.0f}")
        results["prefill"][str(n)] = {"ms": round(sec * 1000, 2), "tok_s": round(tps)}


def bench_decode(model, tokenizer, manager, prompt_ids, results):
    print("\n=== 2. decode (per step, sequences held mid-generation) ===")
    runner = DecodeGraphRunner(model, manager, max_batch=max(DECODE_BATCHES),
                               max_blocks_per_seq=4)
    print("[capture] recording decode graphs ...", flush=True)
    runner.capture()

    print(f"{'batch':>6}  {'eager ms':>9}  {'graph ms':>9}  {'speedup':>8}"
          f"  {'eager tok/s':>12}  {'graph tok/s':>12}")
    for batch in DECODE_BATCHES:
        sched = Scheduler(manager, max_num_seqs=64, max_num_batched_tokens=65536)
        eager = LLMEngine(model, manager, sched)
        seqs = prime(eager, sched, manager, prompt_ids, batch)

        sched_g = Scheduler(manager, max_num_seqs=64, max_num_batched_tokens=65536)
        graphed = LLMEngine(model, manager, sched_g, graph_runner=runner)

        ms_e = timed(lambda: eager.forward_logits(seqs, False), DECODE_STEPS) * 1000
        ms_g = timed(lambda: graphed.forward_logits(seqs, False), DECODE_STEPS) * 1000
        print(f"{batch:6d}  {ms_e:9.2f}  {ms_g:9.2f}  {ms_e / ms_g:7.2f}x"
              f"  {batch / ms_e * 1000:12.0f}  {batch / ms_g * 1000:12.0f}")
        results["decode"][str(batch)] = {
            "eager_ms": round(ms_e, 2), "graph_ms": round(ms_g, 2),
            "eager_tok_s": round(batch / ms_e * 1000),
            "graph_tok_s": round(batch / ms_g * 1000),
        }
        for seq in seqs:
            manager.deallocate(seq)
    return runner


def bench_vs_hf(model, tokenizer, manager, runner, prompt, results):
    print(f"\n=== 3. end to end vs HuggingFace ({E2E_NEW_TOKENS} new tokens) ===")
    prompt_ids = tokenizer(prompt).input_ids

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()

    print(f"{'batch':>6}  {'HF s':>7}  {'ours s':>7}  {'speedup':>8}"
          f"  {'HF tok/s':>9}  {'ours tok/s':>11}")
    for batch in E2E_BATCHES:
        ids = torch.tensor([prompt_ids] * batch, device="cuda")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            hf.generate(ids, max_new_tokens=E2E_NEW_TOKENS,
                        min_new_tokens=E2E_NEW_TOKENS, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        hf_s = time.perf_counter() - t0

        sched = Scheduler(manager, max_num_seqs=64, max_num_batched_tokens=65536)
        eng = LLMEngine(model, manager, sched, graph_runner=runner)
        for _ in range(batch):
            eng.add_request(list(prompt_ids), E2E_NEW_TOKENS, None)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.run_all()
        torch.cuda.synchronize()
        ours_s = time.perf_counter() - t0

        total = batch * E2E_NEW_TOKENS
        print(f"{batch:6d}  {hf_s:7.2f}  {ours_s:7.2f}  {hf_s / ours_s:7.2f}x"
              f"  {total / hf_s:9.0f}  {total / ours_s:11.0f}")
        results["vs_hf"][str(batch)] = {
            "hf_s": round(hf_s, 3), "ours_s": round(ours_s, 3),
            "speedup": round(hf_s / ours_s, 2),
            "hf_tok_s": round(total / hf_s), "ours_tok_s": round(total / ours_s),
        }

    del hf
    torch.cuda.empty_cache()


def bench_prefix(model, tokenizer, filler, results):
    print("\n=== 4. prefix caching (3 prompts sharing a 512-token prefix) ===")
    prefix = (filler * (-(-512 // len(filler))))[:512]
    prompts = [prefix + tokenizer(t).input_ids for t in
               (" In summary,", " Therefore,", " Finally,")]

    manager = build_manager_for(model, num_blocks=64, block_size=256)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generate_many(model, manager, prompts, 16, None)
    torch.cuda.synchronize()
    cold = time.perf_counter() - t0
    cold_rate = manager.hit_rate

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generate_many(model, manager, prompts, 16, None)
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0

    print(f"  cold {cold:.3f}s (hit rate {cold_rate:.0%})")
    print(f"  warm {warm:.3f}s (hit rate {manager.hit_rate:.0%})  "
          f"-> {cold / warm:.2f}x")
    results["prefix_cache"] = {
        "cold_s": round(cold, 3), "warm_s": round(warm, 3),
        "speedup": round(cold / warm, 2),
        "hit_rate": round(manager.hit_rate, 4),
    }


def main():
    tokenizer, model = load_ours()
    filler = tokenizer("Paged attention splits the key-value cache into fixed "
                       "blocks so memory stops fragmenting. ").input_ids
    prompt_ids = tokenizer("The specialty of Hanoi is").input_ids

    results = {"model": MODEL, "gpu": torch.cuda.get_device_name(0),
               "prefill": {}, "decode": {}, "vs_hf": {}}

    manager = build_manager_for(model, num_blocks=200, block_size=256)
    print(f"[setup] {torch.cuda.get_device_name(0)}, "
          f"KV {manager.bytes_per_token() / 1024:.0f} KB/token, "
          f"{manager.capacity_tokens()} token capacity")

    bench_prefill(model, tokenizer, manager, filler, results)
    runner = bench_decode(model, tokenizer, manager, prompt_ids, results)
    bench_vs_hf(model, tokenizer, manager, runner, "The specialty of Hanoi is", results)
    bench_prefix(model, tokenizer, filler, results)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "throughput.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
