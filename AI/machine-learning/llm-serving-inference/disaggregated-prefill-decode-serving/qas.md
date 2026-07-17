# Disaggregated Prefill-Decode Serving - Q&A

---

## Intuition

**Q: What is prefill?**
A: Processing prompt tokens to build the KV cache.

**Q: What is decode?**
A: Autoregressively generating output tokens one step at a time.

**Q: What does disaggregation mean?**
A: Running prefill and decode on separate worker pools.

---

## Motivation

**Q: Why can colocated prefill and decode be bad?**
A: Long prefill jobs can interfere with decode steps and increase output-token latency.

**Q: Which metric is most tied to prefill?**
A: Time to first token, or TTFT.

**Q: Which metric is most tied to decode?**
A: Inter-token latency or time per output token.

---

## Tradeoffs

**Q: What extra cost does disaggregation add?**
A: KV cache transfer from prefill workers to decode workers.

**Q: When can disaggregation hurt?**
A: When transfer overhead or extra queueing exceeds the scheduling benefit.

**Q: Why can separate pools improve serving efficiency?**
A: Each phase can use a batching and parallelism strategy suited to its bottleneck.

---

## Capacity

**Q: What workload needs more prefill capacity?**
A: Long prompts with short outputs.

**Q: What workload needs more decode capacity?**
A: Long generations.

**Q: What should an interviewer hear?**
A: The optimal split is SLO- and workload-dependent, not fixed.
