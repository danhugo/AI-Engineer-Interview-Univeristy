# LoRA Adapter Hot-Swap Overhead - Interview Knowledge Sheet

## Intuition

LoRA serving keeps one base model loaded and applies small task-specific adapters per request.

Hot-swapping means changing which adapter is active without restarting or reloading the base model.

The benefit is multi-tenant specialization. The cost is adapter memory, transfer time, routing complexity, and sometimes reduced batching efficiency.

---

## 1. Adapter Size

For a linear layer:

$$
W \in \mathbb{R}^{d_{out} \times d_{in}}
$$

LoRA uses:

$$
\Delta W = BA
$$

where:

$$
A \in \mathbb{R}^{r \times d_{in}}, \quad B \in \mathbb{R}^{d_{out} \times r}
$$

Adapter parameters for that layer:

$$
r(d_{in} + d_{out})
$$

Total adapter bytes:

$$
\left(\sum_{\ell} r_\ell(d_{in,\ell} + d_{out,\ell})\right)
\cdot \text{bytes per parameter}
$$

---

## 2. Hot-Swap Cost

If an adapter is already resident on GPU, the overhead is mostly routing and extra LoRA compute.

If it is only in CPU memory or on disk, the system must load it:

$$
\text{load time} \ge
\frac{\text{adapter bytes}}{\text{transfer bandwidth}}
$$

This is a lower bound. Real hot-swaps can also pay for deserialization, validation, allocation, synchronization, and cache updates.

---

## 3. GPU Adapter Cache

Serving systems often keep a limited number of adapters on GPU.

Common knobs include:

- maximum active LoRA adapters
- maximum adapter rank
- CPU adapter cache size
- preloaded adapters
- dynamic load and unload endpoints

If a request uses an adapter that is not in the GPU cache, it may pay a cold-load penalty.

---

## 4. Batching Impact

Requests using different adapters can share the base model computation, but their LoRA updates differ.

Efficient LoRA serving needs kernels and scheduling that handle heterogeneous adapters without padding every adapter to the worst case.

If too many unique adapters appear in the same scheduling window, the system may lose batching efficiency or increase memory pressure.

---

## 5. Operational Tradeoffs

Preloading adapters reduces first-request latency but consumes GPU memory that could otherwise hold KV cache.

Dynamic hot-swapping improves flexibility but introduces:

- cold-start latency
- cache eviction decisions
- versioning and authorization concerns
- observability needs for adapter hit rate and load time

A good interview answer separates resident-adapter overhead from cold-adapter overhead.

---

## Interview Gotchas

- LoRA hot-swapping does not reload the base model.
- Adapter rank controls both memory and compute overhead.
- `max_lora_rank` set too high can waste preallocated memory.
- GPU adapter cache misses can add visible latency.
- More adapter memory can reduce room for KV cache.
- Per-request adapter routing must be part of batching and scheduling.

---

## References

- vLLM LoRA adapters documentation: https://docs.vllm.ai/en/stable/features/lora/
- S-LoRA, "Serving Thousands of Concurrent LoRA Adapters": https://arxiv.org/abs/2311.03285
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models": https://openreview.net/forum?id=nZeVKeeFYf9
