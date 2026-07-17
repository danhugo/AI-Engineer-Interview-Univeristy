# Speculative Decoding Acceptance Rate vs Temperature - Q&A

---

## Intuition

**Q: What makes speculative decoding fast?**
A: Accepting multiple draft tokens for each target-model verification pass.

**Q: What does the target model do?**
A: It verifies the drafted tokens and preserves the target distribution.

**Q: What does acceptance rate measure?**
A: How often proposed draft tokens survive verification.

---

## Acceptance

**Q: What is the one-token acceptance probability for proposed token `x`?**
A: $\min(1, p(x) / q(x))$, where `p` is target probability and `q` is draft probability.

**Q: What is the expected one-token acceptance probability?**
A: $\sum_x \min(p(x), q(x))$.

**Q: What does that formula mean intuitively?**
A: Acceptance is high when target and draft distributions overlap.

---

## Temperature

**Q: Does lower temperature always improve acceptance?**
A: No. It helps if both models agree on top tokens, but hurts if their peaks differ.

**Q: What can high temperature do?**
A: It flattens distributions, which may increase one-token overlap but can make long drafts less reliable.

**Q: Why does longer speculation amplify mistakes?**
A: One rejection stops the accepted prefix, so later tokens only matter if earlier tokens were accepted.

---

## Metrics

**Q: What metric is better than raw acceptance rate?**
A: Average accepted tokens per target verification step.

**Q: When is speculation least useful?**
A: When acceptance is low, draft cost is high, or the target GPU is already throughput-saturated.
