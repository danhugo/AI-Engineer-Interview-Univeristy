# PagedAttention KV Cache Utilization - Q&A

---

## Basics

**Q: What problem does PagedAttention address?**
A: KV cache memory fragmentation and over-reservation during LLM serving.

**Q: What is the core idea?**
A: Store each sequence's KV cache in fixed-size blocks that can live in non-contiguous physical memory.

**Q: Does PagedAttention change the attention result?**
A: No. It changes memory layout and access, not the dense attention math.

---

## Utilization

**Q: How many blocks are needed for `T` tokens and block size `b`?**
A: `ceil(T / b)`.

**Q: Where does internal waste happen?**
A: In the final partially filled block of each sequence.

**Q: What is single-sequence utilization?**
A: `used_tokens / (ceil(used_tokens / block_size) * block_size)`.

---

## Design

**Q: Why not use very large blocks?**
A: Large blocks waste more tail space for variable-length sequences.

**Q: Why not use extremely tiny blocks?**
A: Tiny blocks increase metadata and kernel/scheduling overhead.

**Q: What is the block table?**
A: A mapping from a sequence's logical blocks to physical KV blocks.

---

## Gotchas

**Q: Does PagedAttention reduce model weight memory?**
A: No. It targets dynamic KV cache memory.

**Q: Why does better KV utilization improve throughput?**
A: More requests can fit in GPU memory, enabling larger active batches.

**Q: How does this relate to prefix caching?**
A: Prefix caching can reuse full KV blocks managed by the same block-based cache system.
