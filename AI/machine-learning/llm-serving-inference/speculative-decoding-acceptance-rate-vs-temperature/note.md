# Speculative Decoding Acceptance Rate vs Temperature - Interview Knowledge Sheet

## Intuition

Speculative decoding is fast when draft tokens are accepted.

The draft model proposes several tokens. The target model verifies them in parallel. If the draft distribution is close to the target distribution, many tokens are accepted per target forward pass.

Temperature matters because it changes both distributions:

$$
p_T(x) = \text{softmax}(z_p / T)
$$

$$
q_T(x) = \text{softmax}(z_q / T)
$$

where `p` is the target distribution and `q` is the draft distribution.

---

## 1. Acceptance Rule

For one proposed token `x` sampled from draft distribution `q`, speculative sampling accepts with probability:

$$
\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right)
$$

The expected one-token acceptance probability is:

$$
\sum_x q(x)\min\left(1, \frac{p(x)}{q(x)}\right)
= \sum_x \min(p(x), q(x))
$$

So acceptance is high when the two distributions overlap.

---

## 2. Effect of Temperature

Low temperature sharpens distributions. If both draft and target put most mass on the same token, acceptance can rise.

But if they disagree on the top token, low temperature can make acceptance worse because the distributions become very different.

High temperature flattens distributions. This may increase overlap in some cases, but it also makes draft samples more random and can lower long-prefix acceptance.

The right statement is:

$$
\text{acceptance depends on distribution overlap after sampling controls}
$$

not simply "lower temperature always accepts more."

---

## 3. Prefix Acceptance

If `k` speculative tokens are proposed, a useful metric is average accepted tokens per verification step.

With independent per-position acceptance approximation:

$$
E[\text{accepted}] \approx \sum_{i=1}^{k} \prod_{j=1}^{i} a_j
$$

where `a_j` is the probability that token `j` is accepted given the previous draft prefix was accepted.

One early rejection stops the speculative prefix.

---

## 4. Speedup Intuition

Speculation helps when:

- draft generation is cheap
- target verification of multiple tokens is close to one normal target step
- acceptance rate is high
- batch size is not already saturating the GPU

Low acceptance wastes draft work and verifier slots.

---

## Interview Gotchas

- Acceptance is about overlap between target and draft distributions.
- Temperature changes the distributions before acceptance is computed.
- Greedy matching is not the same as lossless speculative sampling.
- Long speculative lengths amplify rejection risk.
- Measure accepted tokens per target forward pass, not only per-token acceptance.

---

## References

- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding": https://proceedings.mlr.press/v202/leviathan23a.html
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling": https://arxiv.org/abs/2302.01318
- vLLM speculative decoding docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/
