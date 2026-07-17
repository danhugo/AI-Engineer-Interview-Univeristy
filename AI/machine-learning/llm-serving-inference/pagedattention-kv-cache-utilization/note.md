# PagedAttention KV Cache Utilization - Interview Knowledge Sheet

## Intuition

PagedAttention treats KV cache memory like virtual memory pages.

Instead of reserving one large contiguous KV region for each request, it splits every sequence into fixed-size blocks. Logical token blocks map to physical memory blocks that do not need to be contiguous.

This improves utilization because memory is allocated as tokens arrive.

---

## 1. The Fragmentation Problem

Without paging, a serving system may reserve memory for each request's maximum possible sequence length.

If a request reserves 2048 tokens but only reaches 300 tokens, most of that reservation is wasted.

For variable-length traffic, this can make GPU memory look full even though many reserved token slots are unused.

---

## 2. Block Allocation

PagedAttention partitions each sequence into blocks of `block_size` tokens.

The number of physical blocks needed for one sequence of length $T$ is:

$$
\left\lceil \frac{T}{\text{block size}} \right\rceil
$$

Allocated token slots:

$$
\left\lceil \frac{T}{\text{block size}} \right\rceil \cdot \text{block size}
$$

Only the final block can be partially empty.

---

## 3. Utilization

For one sequence:

$$
\text{utilization} =
\frac{\text{used token slots}}{\text{allocated token slots}}
$$

For a batch:

$$
\text{batch utilization} =
\frac{\sum_i T_i}{\sum_i \left\lceil T_i / b \right\rceil b}
$$

where $b$ is the block size.

Smaller blocks reduce internal waste but can increase metadata, scheduling, and kernel overhead.

---

## 4. Why It Helps Serving

PagedAttention enables:

- on-demand KV allocation
- non-contiguous physical KV blocks
- better memory sharing for parallel sampling and beam search
- more active requests in the same GPU memory budget
- cleaner eviction and prefix-cache block management

The serving win is often higher throughput from larger batches, not a change in transformer math.

---

## 5. Relation to OS Paging

The analogy:

| OS concept | PagedAttention concept |
| --- | --- |
| process virtual address space | sequence logical blocks |
| physical pages | physical KV blocks |
| page table | block table |
| internal fragmentation in last page | unused slots in last KV block |

The attention kernel uses the block table to fetch the right KV blocks.

---

## Interview Gotchas

- PagedAttention does not make dense attention subquadratic.
- It improves KV memory management and reduces fragmentation.
- Only the last block of a sequence needs to be partially empty.
- Larger blocks waste more tail space; smaller blocks add overhead.
- Utilization should be measured over allocated KV slots, not model weights.
- Prefix caching and PagedAttention fit naturally because both operate on KV blocks.

---

## References

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention": https://arxiv.org/abs/2309.06180
- vLLM PagedAttention blog: https://vllm-project.github.io/2023/06/20/vllm.html
- vLLM paged attention design note: https://docs.vllm.ai/en/stable/design/paged_attention/
