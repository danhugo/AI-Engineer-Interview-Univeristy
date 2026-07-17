# Break-Even Pay-Per-Token API vs Dedicated GPU - Q&A

---

## Intuition

**Q: What is the core break-even comparison?**
A: Pay-per-token API cost versus dedicated serving cost over the same token volume.

**Q: Why can APIs be cheaper at low volume?**
A: You pay only for usage and avoid idle GPU time.

**Q: Why can dedicated GPUs be cheaper at high volume?**
A: Fixed hourly cost is amortized over many tokens.

---

## API Cost

**Q: What inputs are needed for API cost?**
A: Input tokens, output tokens, and their prices per million tokens.

**Q: Why separate input and output tokens?**
A: They often have different prices.

**Q: What can lower API cost?**
A: Prompt caching, batch discounts, smaller models, and shorter outputs.

---

## GPU Cost

**Q: What inputs are needed for GPU cost per token?**
A: Hourly cost, token throughput, and utilization.

**Q: Why is utilization central?**
A: Idle GPU time is still billed.

**Q: What non-GPU costs matter?**
A: CPUs, memory, storage, networking, engineering, monitoring, and on-call operations.

---

## Decision

**Q: What is a common bad comparison?**
A: API list price versus GPU rental price without considering utilization and operations.

**Q: What workload favors dedicated GPUs?**
A: Predictable, high-volume, batchable traffic with acceptable self-hosted model quality.

**Q: What workload favors APIs?**
A: Bursty traffic, low volume, fast product iteration, or need for managed model quality and features.
