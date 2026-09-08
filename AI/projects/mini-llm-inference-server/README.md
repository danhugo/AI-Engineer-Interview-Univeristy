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

Each stage is runnable on its own.

1. **Skeleton** — Qwen3 forward with `flash-attn`, 1 GPU, one prompt generates.
2. **Paged KV cache** — `BlockManager` + Triton KV-store kernel; attention reads block tables.
3. **Scheduler + engine loop** — continuous batching over many prompts (offline).
4. **Prefix caching** — hash blocks, dedup, ref-count.
5. **Tensor parallelism (TP=2)** — shard weights, NCCL, process spawn.
6. **CUDA graphs** — capture/replay decode for fixed batch buckets.
7. **Flash-attention study kernel** — Triton, tested to match `flash-attn`.
8. **Serving API** — OpenAI-compatible endpoint + streaming.
9. **Benchmark** — prefill / decode tokens-per-second.

## Correctness anchor

`test_correctness.py` runs greedy decode and asserts the output matches
`transformers` for the same prompt — this catches paging and TP bugs.
`test_flash_attn_study.py` asserts the study kernel matches `flash-attn`
within tolerance.
