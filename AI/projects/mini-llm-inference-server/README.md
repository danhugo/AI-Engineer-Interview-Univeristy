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
                 ┌────────▼────────┐
                 │   Serving API   │  FastAPI, /v1/chat/completions, SSE streaming
                 └────────┬────────┘
                          │  add_request
                 ┌────────▼────────┐
                 │    LLMEngine    │  the step() loop: schedule → run → postprocess
                 └────┬───────┬────┘
          schedule    │       │   run(seqs, is_prefill)
              ┌───────▼──┐  ┌─▼──────────────┐
              │Scheduler │  │  ModelRunner   │  tensor prep, CUDA graphs, TP driver
              │(batching)│  └─┬────────────┬─┘
              └────┬─────┘    │            │
                   │          │        ┌───▼────┐
            ┌──────▼──────┐   │        │ Sampler│  temperature / greedy / top-k,p
            │ BlockManager│◄──┘        └────────┘
            │ paged KV +  │   │
            │ prefix cache│   ▼
            └─────────────┘  Model (Qwen3, TP=2 sharded)
                   ▲          │
                   │      ┌───▼──────────────────────┐
              Sequence    │ Attention layer          │
              (per-req    │ flash_attn_varlen (prefill)
               state)     │ flash_attn_w_kvcache (dec)│
                          │ + Triton KV-store kernel  │
                          └───────────────────────────┘

  Study-only (not imported by the engine):
     Flash-attn study kernel  ── Triton, tested to match flash-attn
  Tooling:
     Benchmark  ── prefill / decode tokens-per-second
```

### Blocks

| Block | Responsibility |
|---|---|
| **Serving API** | OpenAI-compatible HTTP endpoint + SSE streaming; drives the engine loop |
| **LLMEngine** | Request queue + `step()` = schedule → run → postprocess |
| **Scheduler** | Continuous batching: waiting/running queues, prefill-priority, preemption, chunked prefill |
| **Sequence** | Per-request state: tokens, block table, status |
| **BlockManager** | Paged KV cache: block pool, `slot_mapping`, prefix caching (hash + ref-count) |
| **ModelRunner** | Flatten sequences → tensors, run forward, capture/replay CUDA graphs, drive TP |
| **Model (Qwen3)** | The transformer, weights sharded across 2 GPUs |
| **Attention layer** | `flash-attn` for prefill/decode + a Triton kernel that writes K/V into cache blocks |
| **Sampler** | Temperature + greedy (top-k / top-p as stretch) |
| **Tensor Parallelism** | `torch.multiprocessing` spawn + NCCL all-reduce; TP=2 |
| **CUDA graphs** | Record decode step once, replay it — removes launch overhead |
| **Flash-attn study kernel** | Hand-written Triton flash attention; study-only, tested vs `flash-attn` |
| **Benchmark** | Prefill / decode throughput vs raw HuggingFace |

## Build stages

All nine done. Each has a test that gates it.

| | Stage | Test | Gate |
|---|---|---|---|
| 1 | Qwen3 forward with `flash-attn` | `run.py`, `diag_fp32.py` | fp32 bit-identical to HF |
| 2 | Paged KV cache | `test_stage2.py` | cached output == uncached, token for token |
| 3 | Scheduler + continuous batching | `test_stage3.py` | batched logits == unbatched |
| 4 | Prefix caching | `test_stage4.py` | 512 tokens reused, answer unchanged |
| 5 | Tensor parallelism TP=2 | `test_stage5.py` | TP=2 logits == TP=1 |
| 6 | CUDA graphs | `test_stage6.py` | **bit-exact**, 2x faster |
| 7 | Flash-attn kernel (study) | `test_stage7.py` | matches `flash-attn` accuracy |
| 8 | OpenAI-compatible server | `test_stage8.py` | streaming + 4.8x on 8 concurrent |
| 9 | Benchmark | `bench/throughput.py` | see Results |

Run any of them with `./sync.sh py test_stage4.py`.

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

### Tensor parallelism

| | heads/rank | kv_heads/rank | KV per token | params/rank |
|---|---|---|---|---|
| TP=1 | 32 | 8 | 144 KB | 8.19B |
| TP=2 | 16 | 4 | **72 KB** | 4.72B |

Halving KV bytes per rank is the real win: cache capacity scales with GPUs.

## Correctness anchor

`test_correctness.py` runs greedy decode and asserts the output matches
`transformers` for the same prompt — this catches paging and TP bugs.
`test_flash_attn_study.py` asserts the study kernel matches `flash-attn`
within tolerance.
