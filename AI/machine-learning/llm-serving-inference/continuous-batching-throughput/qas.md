# Continuous Batching Throughput - Q&A

---

## Basics

**Q: What is continuous batching?**
A: A scheduling strategy that reforms LLM batches at iteration boundaries so new requests can enter as old ones finish.

**Q: What is another name for continuous batching?**
A: In-flight batching or iteration-level batching.

**Q: Why is request-level batching inefficient for LLM generation?**
A: Requests have different output lengths, so short requests finish while long requests keep the batch alive.

---

## Throughput

**Q: Why can continuous batching increase TPS?**
A: It keeps more batch slots occupied across decode iterations.

**Q: What is a simple decode TPS approximation?**
A: `active_sequences / decode_step_time_seconds`.

**Q: Does continuous batching guarantee lower latency for every request?**
A: No. It improves utilization, but queueing and interference can still hurt latency.

---

## Prefill and Decode

**Q: Why is prefill special?**
A: It processes the whole prompt before decode can produce streamed output.

**Q: What problem can long prefills cause?**
A: They can delay decode iterations and worsen ITL.

**Q: Name one mitigation for prefill/decode interference.**
A: Chunked prefill, scheduling limits, or disaggregated prefill/decode workers.

---

## Limits

**Q: What memory structure often limits active batch size?**
A: The KV cache.

**Q: What happens when KV cache is full?**
A: New requests must wait, even if compute is otherwise available.

**Q: Why does fairness matter?**
A: Without fairness, some request classes can be starved or see poor tail latency.
