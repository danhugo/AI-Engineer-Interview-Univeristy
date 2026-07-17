# KV Cache Tiered Offloading - Interview Knowledge Sheet

## Intuition

The KV cache can become the largest dynamic memory consumer during long-context inference.

Tiered offloading keeps the hottest KV cache blocks in fast GPU memory and moves colder blocks to slower, larger memory such as CPU RAM or SSD.

The tradeoff is simple:

$$
\text{more capacity} \leftrightarrow \text{more data movement latency}
$$

---

## 1. Why Offload KV Cache

Autoregressive decoding stores key and value vectors for every previous token.

A simplified estimate is:

$$
\text{KV bytes} =
2 \cdot L \cdot B \cdot T \cdot H \cdot \text{bytes per element}
$$

where:

- `2`: key and value
- $L$: layers
- $B$: batch size
- $T$: cached tokens per sequence
- $H$: hidden size across KV heads

As $T$ and $B$ grow, KV cache can limit batch size before compute does.

---

## 2. Memory Tiers

Typical tiers:

| Tier | Capacity | Latency | Bandwidth | Use |
| --- | ---: | ---: | ---: | --- |
| GPU HBM | lowest | lowest | highest | active decode blocks |
| CPU RAM | larger | higher | lower | inactive or older blocks |
| SSD | largest | highest | lowest | overflow or batch-level staging |

A serving system wants GPU memory to hold the KV blocks needed soon.

---

## 3. Transfer Cost

Moving bytes across tiers has a minimum bandwidth cost:

$$
\text{transfer seconds} =
\frac{\text{bytes moved}}{\text{bandwidth bytes/sec}}
$$

Real systems also pay scheduling, DMA setup, page pinning, and synchronization overhead.

If a decode step needs an offloaded block and it was not prefetched, the model may stall while waiting for the transfer.

---

## 4. Prefetching

Offloading is most useful when the system can predict which KV blocks will be needed.

For standard left-to-right attention, the newest decode token attends over prior tokens, so older KV blocks may be needed repeatedly. Some systems rely on chunking, sliding windows, attention sparsity, or request scheduling to make offload practical.

The prefetch rule is:

$$
\text{transfer time} \le \text{available compute time before use}
$$

If this is true, data movement can be hidden. If not, offloading reduces memory pressure but increases latency.

---

## 5. What To Offload

Common policies:

- keep active request tails on GPU
- offload paused or low-priority requests
- evict least-recently-used blocks
- offload older context when attention pattern allows it
- use CPU RAM as a staging cache before SSD

The best policy depends on workload shape: long context, many concurrent requests, streaming outputs, and whether requests pause or resume.

---

## Interview Gotchas

- Offloading solves capacity pressure, not compute cost.
- Decode can become transfer-bound if needed KV blocks are off GPU.
- Bandwidth math gives a lower bound; real latency is higher.
- Keeping more KV cache on GPU often increases throughput by enabling larger active batches.
- Quantizing KV cache and offloading KV cache are different optimizations and can be combined.
- Offloading weights, activations, and KV cache have different access patterns.

---

## References

- FlexGen, "High-throughput Generative Inference of Large Language Models with a Single GPU": https://arxiv.org/abs/2303.06865
- Hugging Face Transformers KV cache documentation: https://huggingface.co/docs/transformers/main/cache_explanation
- vLLM PagedAttention blog, KV cache memory discussion: https://vllm-project.github.io/2023/06/20/vllm.html
