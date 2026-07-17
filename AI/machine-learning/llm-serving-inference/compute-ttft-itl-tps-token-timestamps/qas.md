# Compute TTFT, ITL, TPS from Token Timestamps - Q&A

---

## Metrics

**Q: How do you compute TTFT from token timestamps?**
A: `first_token_time - request_start_time`.

**Q: How do you compute average ITL for one request?**
A: `(last_token_time - first_token_time) / (num_tokens - 1)`.

**Q: When is ITL undefined?**
A: When the response has fewer than two output tokens.

---

## Throughput

**Q: How do you compute per-request TPS?**
A: `output_tokens / end_to_end_latency`.

**Q: How do you compute aggregate benchmark TPS?**
A: Total output tokens divided by the benchmark response window.

**Q: Why can per-user TPS and aggregate TPS move in opposite directions?**
A: More concurrency can increase total work completed while increasing per-request waiting time.

---

## Timestamping

**Q: Why use a monotonic clock?**
A: It cannot jump backward or forward due to wall-clock adjustments.

**Q: Should empty stream events count as tokens?**
A: No. Count only actual output tokens.

**Q: Why might client-side TTFT be higher than server-side TTFT?**
A: Client-side timing includes network, serialization, and client processing overhead.

---

## Gotchas

**Q: Should TTFT be included in ITL?**
A: Usually no; many serving benchmarks define ITL over gaps after the first token.

**Q: What percentile is often checked for user-facing latency?**
A: P95 or P99, because tail latency drives bad user experiences.

**Q: What timestamp should mark completion for streaming?**
A: Use a consistent choice, usually the final content token or final done event.
