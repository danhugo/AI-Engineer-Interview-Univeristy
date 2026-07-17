# Tokens Per Second Throughput - Q&A

---

## Basics

**Q: What does TPS usually mean in LLM serving?**
A: Tokens per second, usually generated output tokens per second unless otherwise stated.

**Q: Why should you ask which tokens are counted?**
A: Prompt tokens and generated tokens exercise different phases of inference.

**Q: What is aggregate output TPS?**
A: Total generated tokens across all requests divided by the measurement window.

---

## User Experience

**Q: Can aggregate TPS improve while user latency worsens?**
A: Yes. Larger batches can improve GPU utilization while increasing queueing or per-token delay.

**Q: What is per-user TPS?**
A: Output tokens for one request divided by that request's end-to-end latency.

**Q: What metric does long-output per-user TPS approach?**
A: Approximately `1 / ITL`.

---

## Benchmarking

**Q: Why does the benchmark time window matter?**
A: Including setup, warmup, or result-writing overhead lowers measured TPS.

**Q: What should be excluded from steady-state TPS?**
A: Warmup and cooldown requests, unless cold-start performance is the target.

**Q: Why are TPS numbers hard to compare across tokenizers?**
A: Different tokenizers split the same text into different token counts.

---

## Bottlenecks

**Q: What bottleneck often limits prefill throughput?**
A: GPU compute.

**Q: What bottleneck often limits decode throughput?**
A: Memory bandwidth and KV cache reads.

**Q: What happens when concurrency exceeds useful capacity?**
A: Queueing grows and latency worsens, with little or no TPS gain.
