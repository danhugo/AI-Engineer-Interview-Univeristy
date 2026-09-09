# Mini LLM Inference Server

A small, readable LLM inference server. Built to **learn the serving layer**:
paged KV cache, continuous batching, flash attention, and tensor parallelism.

Reference implementations: [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
and [mini-vllm](https://github.com/jianzhnie/mini-vllm).

## Goal

Serve a real SLM (Qwen3, a few billion params) across **2 GPUs** with:

- **Paged KV cache** — no memory fragmentation, shareable prefixes.
- **Continuous batching** — schedule per token-step, not per request.
- **Flash attention** — via the `flash-attn` library, plus our own study kernel.
- **Tensor parallelism (TP=2)** — shard the model across both GPUs.
- **CUDA graphs** — kill kernel-launch overhead in the decode phase.
- **OpenAI-compatible API** — with token streaming.

Repo rule (same as the rest of this course): **from-scratch code is for study**
(it gets a test proving it matches the library). **The real engine uses libraries.**
The one study module here is the flash-attention Triton kernel.

## Architecture

```
                  HTTP request (OpenAI format)
                             │
                  ┌──────────▼──────────┐
                  │  server/api.py      │  FastAPI, SSE streaming.
                  │  AsyncEngine        │  Engine runs on its own thread;
                  └──────────┬──────────┘  handlers never touch the model.
                             │  add_request → asyncio.Queue per request
                  ┌──────────▼──────────┐
                  │ engine/llm_engine   │  step() = schedule → run → postprocess
                  │ LLMEngine           │  also builds the flat input tensors
                  └────┬───────────┬────┘
            schedule()  │           │  forward_logits(seqs, is_prefill)
              ┌─────────▼──┐   ┌────▼──────────────┐
              │ scheduler  │   │ models/qwen3.py   │  36 x DecoderLayer
              │ Scheduler  │   │ Qwen3ForCausalLM  │
              └─────┬──────┘   └────┬─────────┬────┘
   allocate/preempt │               │         │
              ┌─────▼────────┐      │    ┌────▼──────────┐
              │ block_manager│◄─────┘    │ layers/linear │  Column/Row
              │ BlockManager │  slot_     │               │  parallel (TP)
              │ paged KV +   │  mapping   └────┬──────────┘
              │ prefix cache │                 │ all_reduce
              └─────┬────────┘            ┌────▼──────────┐
                    │                     │utils/parallel │  NCCL, SPMD
              ┌─────▼────────┐            └───────────────┘
              │ sequence.py  │
              │ Sequence     │       ┌────────────────────────────┐
              └──────────────┘       │ layers/attention.py        │
                                     │ varlen (prefill) /         │
              ┌──────────────┐       │ with_kvcache (decode),     │
              │utils/context │──────►│ both via block_table       │
              │ per-step     │       └────────────────────────────┘
              │ slot_mapping │
              │ cu_seqlens   │       ┌────────────────────────────┐
              │ block_table  │       │ engine/cuda_graph.py       │
              └──────────────┘       │ DecodeGraphRunner          │
                                     │ bucketed replay, sink block│
              ┌──────────────┐       └────────────────────────────┘
              │layers/sampler│
              │ temp/top-p/k │
              └──────────────┘

  Study only — the engine never imports this:
      layers/flash_attn_study.py   Triton flash attention, tiled online softmax

  Tooling:
      bench/throughput.py    prefill / decode tok/s, graphs on-off, vs HF
      bench/intelligence.py  GSM8K / MATH-500, ours vs HF and vs published
      diag_layers.py         where does divergence start, per layer
      diag_batch.py          is a divergence a real bug or a near-tie
```

### Files

| File | Responsibility |
|---|---|
| `server/api.py` | OpenAI-compatible endpoints, SSE streaming, incremental detokenisation. `AsyncEngine` puts the engine on a background thread |
| `engine/llm_engine.py` | `step()` = schedule → run → postprocess. Flattens sequences into tensors, runs the forward, samples |
| `engine/scheduler.py` | Continuous batching: waiting/running deques, prefill-priority, recompute preemption |
| `engine/sequence.py` | One request: tokens, block table, sampling params, finish reason |
| `engine/block_manager.py` | Paged KV cache: block pool, `slot_mapping`, prefix caching (chained hash + ref-count), sink block |
| `engine/cuda_graph.py` | `DecodeGraphRunner`: one graph per bucketed batch size, static buffers |
| `engine/generate.py` | Offline convenience wrappers over the engine |
| `models/qwen3.py` | The transformer: GQA + QK-Norm + RoPE + SwiGLU. Module names match HuggingFace |
| `layers/attention.py` | `flash_attn_varlen_func` (prefill) and `flash_attn_with_kvcache` (decode), both reading the block table; plus the K/V scatter |
| `layers/linear.py` | `ColumnParallelLinear` / `RowParallelLinear` for tensor parallelism |
| `layers/sampler.py` | Temperature, top-p, top-k, greedy at temp 0; seeded per step so TP ranks agree |
| `layers/flash_attn_study.py` | **Study only.** Triton flash attention. The engine never imports it |
| `utils/context.py` | Per-step `slot_mapping` / `cu_seqlens` / `block_table`, so the model signature stays free of serving concerns |
| `utils/parallel.py` | torchrun init, `all_reduce`, rank-0 logging |
| `utils/loader.py` | Loads HF weights, slicing them per rank by reading the shard dim off the layer class |
| `common.py` | Shared fixtures: model path, test prompts, loading |

### Deliberately not built

Named here because the first draft of this README promised them:

- **No `ModelRunner` class.** Tensor prep lives in `LLMEngine`, CUDA graphs in
  `DecodeGraphRunner`, TP in `layers/linear.py` + `utils/parallel.py`. Splitting
  them that way meant no extra indirection layer was needed.
- **No Triton kernel for the K/V scatter.** `store_kv` flattens the first two
  cache dims and does one indexed assignment. Plain torch, and not the
  bottleneck. Triton appears only in the study kernel.
- **No chunked prefill.** A step is all-prefill or all-decode. vLLM mixes them
  so a long prompt cannot stall decodes; that is a refinement we skipped.
- **No driver/worker split for TP.** Every rank runs the same engine loop
  (SPMD), so identical logits keep the schedulers in lockstep with no command
  channel. vLLM and nano-vllm both need one; we do not.
- **No vocab-parallel embedding.** `embed_tokens` and `lm_head` are replicated,
  costing ~1.2B params per rank in exchange for skipping an all-gather.

## Build stages

All nine done. Each has a test that gates it.

| | Stage | Test | Gate |
|---|---|---|---|
| 1 | Qwen3 forward with `flash-attn` | `test_stage1.py` | fp32 **exactly** equal to HF, all metrics |
| 2 | Paged KV cache | `test_stage2.py` | cached output == uncached, token for token |
| 3 | Scheduler + continuous batching | `test_stage3.py` | batched logits == unbatched |
| 4 | Prefix caching | `test_stage4.py` | 512 tokens reused, answer unchanged |
| 5 | Tensor parallelism TP=2 | `test_stage5.py` | TP=2 logits == TP=1 |
| 6 | CUDA graphs | `test_stage6.py` | **bit-exact**, 2x faster |
| 7 | Flash-attn kernel (study) | `test_stage7.py` | matches `flash-attn` accuracy |
| 8 | OpenAI-compatible server | `test_stage8.py` | streaming + 4.8x on 8 concurrent |
| 9 | Benchmark | `bench/throughput.py` | see Results |

Run any of them with `./sync.sh py test_stage4.py`. Stage 1 first — everything
after it assumes the model is right, and would happily pass while comparing
ours against ours.

`diag_layers.py` and `diag_batch.py` are diagnostics, not gates: reach for them
when a test fails and you need to know *where* the divergence starts or whether
it is just a near-tie.

## Results

Qwen3-8B, bf16, one A100 80GB. `./sync.sh run 'python -m bench.throughput'`.

### Decode — the whole point of CUDA graphs

| batch | eager ms | graph ms | speedup | eager tok/s | graph tok/s |
|---|---|---|---|---|---|
| 1 | 33.08 | 15.29 | 2.16x | 30 | 65 |
| 4 | 33.26 | 15.71 | 2.12x | 120 | 255 |
| 16 | 33.14 | 16.65 | 1.99x | 483 | 961 |
| 32 | 33.19 | 17.31 | 1.92x | 964 | 1849 |

Read the `eager ms` column: **33.1 ms whether the batch is 1 or 32**. Thirty-two
times the arithmetic in the same wall clock means the GPU is idle waiting on the
CPU to launch kernels. That is why graphs help, and why they help most when
there is least work to hide the launches behind.

### End to end vs HuggingFace `generate()`

| batch | HF s | ours s | speedup | HF tok/s | ours tok/s |
|---|---|---|---|---|---|
| 1 | 3.46 | 1.01 | **3.41x** | 19 | 63 |
| 8 | 2.77 | 1.06 | **2.60x** | 185 | 481 |

64 new tokens, greedy, same prompt.

### Prefill

| prompt tokens | ms | tokens/s |
|---|---|---|
| 128 | 38.5 | 3,322 |
| 512 | 51.4 | 9,962 |
| 2048 | 187.7 | 10,912 |

Throughput saturates near 10k tok/s. The 128-token case looks bad only because
fixed per-step overhead has nothing to amortise against — prefill is
compute-bound, which is exactly why it is not worth graphing.

### Prefix caching

3 prompts sharing a 512-token prefix: cold 0.646s → warm 0.540s (**1.20x**),
hit rate 50%. Modest here because 16 decode steps dominate the total; the win
grows with longer shared prefixes and shorter generations. On prefill alone,
99% of prompt tokens skipped attention compute.

### Quality — does it generate the *right* tokens?

Fast is easy; fast and correct is the point. Two separate checks.

**Implementation gate** — ours vs HuggingFace on the same prompts, greedy:

| task | identical text | same final answer | accuracy ours / HF | time ours / HF |
|---|---|---|---|---|
| GSM8K (40) | 15/40 | **40/40 (100%)** | 87.5% / 87.5% | 20.1s / 128.4s (6.4x) |
| MATH-500 (40) | 11/40 | 32/40 (80%) | 80.0% / 70.0% | 35.4s / 287.9s (8.1x) |

Identical *text* is low and that is expected — exact bf16 ties make the two
trajectories fork (see NOTES.md). Identical *answers* is the meaningful number,
and on GSM8K it is 40/40 with byte-identical accuracy.

**Quality** — ours with Qwen's recommended non-thinking sampling
(`temperature=0.7, top_p=0.8, top_k=20`), 200 problems:

| benchmark | ours | published | note |
|---|---|---|---|
| GSM8K | **95.0%** | 89.84% | published figure is 8B-**Base** 4-shot; instruct non-thinking beating it is expected |
| MATH-500 | **81.5%** | 87.4% | see below |

MATH-500 converges as the output budget grows, which identifies the limit as
truncation rather than a bug:

| max_new | accuracy | answers reaching `\boxed{}` |
|---|---|---|
| 1024 | 70.0% | 76% |
| 2048 | 77.5% | 90% |
| 4096 | 81.5% | 96% |

The remaining ~6 points: 4% of answers still truncate, the string-based math
comparison here is a **floor** (`\frac{3}{56}` needs real symbolic
equivalence, not `==`), sampling adds ±2-3 points, and this is a 200-problem
subset of 500.

Worth noting: fixing the *scorer* moved GSM8K from 88.5% to 95.0%. The model
writes `\$70,000` and `60\%` where the gold answers are bare digits. Six and a
half points of "model quality" were an artefact of the harness.

### Tensor parallelism

| | heads/rank | kv_heads/rank | KV per token | params/rank |
|---|---|---|---|---|
| TP=1 | 32 | 8 | 144 KB | 8.19B |
| TP=2 | 16 | 4 | **72 KB** | 4.72B |

Halving KV bytes per rank is the real win: cache capacity scales with GPUs.

## How this is tested

Paged KV, batching, prefix caching, TP and CUDA graphs are all supposed to be
*bit-neutral* rearrangements: faster, same answer. So every stage is gated on
"did the answer change", never on "is it fast".

**fp32 is only possible at stage 1.** From stage 2 on, every path goes through
flash-attn's paged kernels, which accept fp16/bf16 only. So stage 1 is the one
place exact equality can be demanded — the difference between *proving*
correctness and merely *failing to detect* a problem:

```
fp32, seq 1024:  max|diff| 0.000000   TV 0.000000   top-1 100%   layer-16 0.000000
```

**bf16 needs several metrics, because none of them is trustworthy alone:**

| Metric | Role | Why not alone |
|---|---|---|
| top-k mutual agreement | **the gate** | — (this is vLLM's own check) |
| worst-position TV distance | health number | blind to *where* |
| max abs logit diff | report only | dominated by tail tokens at ~1e-16 probability |
| exact top-1 match | report only | flips on exact ties, with nothing wrong |
| final answers vs HF | **the real gate** | what a user actually sees |

Stage 1's bf16 run is the calibration: it is the only run where fp32 has
already proven the code correct, so the spread there *is* the normal amount
(TV 0.026-0.060). Later stages compare against it.

CUDA graphs are the exception — bf16 but still gated **bit-exact**, since a
replay runs identical kernels in identical order at identical addresses.

### Two traps that cost real time

**The scorer is part of the measurement.** GSM8K read 88.5% until the answer
comparison learned that `\$70,000` and `70000` are the same number. Then 95.0%.
Six and a half points of apparent model quality were a harness artefact.

**Truncation looks exactly like stupidity.** MATH-500 read 70.0% against a
published 87.4. The tell was not the score but that only 76% of answers
contained `\boxed{}` — the rest were cut off mid-derivation. Raising the output
budget: 1024 → 70.0%, 2048 → 77.5%, 4096 → 81.5%, with the completion rate
tracking it at 76% → 90% → 96%.

`NOTES.md` has the full numerical write-up, including why HuggingFace is **not**
ground truth in bf16 (at one element fp32 says 10274.6, HF's bf16 says 8448,
ours says 52.75 — both wrong).
