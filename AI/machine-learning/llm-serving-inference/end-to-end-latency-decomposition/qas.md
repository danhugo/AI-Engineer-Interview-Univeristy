# End-to-End Latency Decomposition - Q&A

---

## Basics

**Q: What is end-to-end latency?**
A: The elapsed time from request start until the client receives completion.

**Q: What are common components of LLM E2E latency?**
A: Network, queueing, tokenization, prefill, decode, detokenization, and streaming.

**Q: Why decompose latency?**
A: Different bottlenecks require different fixes.

---

## Prefill and Decode

**Q: Which phase usually dominates TTFT?**
A: Queueing plus prefill.

**Q: Which phase usually dominates ITL?**
A: Decode iterations.

**Q: Why do long prompts increase TTFT?**
A: The prefill phase must process the prompt before the first output token appears.

**Q: Why do long outputs increase E2E latency?**
A: Each additional output token requires another decode step.

---

## Queueing

**Q: What does rising queue time indicate?**
A: Arrival rate is close to or above useful service capacity.

**Q: Can batching add queueing latency?**
A: Yes. Requests may wait briefly to form larger batches.

**Q: Why can tail latency rise before average latency looks bad?**
A: A subset of requests may wait behind long or unlucky work.

---

## Debugging

**Q: What should you compare alongside latency?**
A: Prompt length, output length, and concurrency.

**Q: What is the rough streaming E2E formula?**
A: `TTFT + (output_tokens - 1) * average_ITL + finish_overhead`.

**Q: Does streaming reduce model compute?**
A: No. It improves perceived responsiveness by sending tokens as they are produced.
