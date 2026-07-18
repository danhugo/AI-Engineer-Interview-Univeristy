# Compute-Optimal Model Size IsoFLOP - Interview Knowledge Sheet

## Intuition

A fixed training compute budget can be spent on a larger model, more data tokens, or both.

Compute-optimal scaling asks:

> For this compute budget, what model size and data size minimize loss?

The simplified dense-transformer training compute approximation is:

$$
C \approx 6ND
$$

where `N` is parameters and `D` is training tokens.

---

## 1. IsoFLOP Curves

An IsoFLOP curve holds compute fixed:

$$
D = \frac{C}{6N}
$$

If `N` is too large, the model is undertrained because `D` is too small. If `N` is too small, the model lacks capacity and gets too many repeated tokens for its size.

---

## 2. Chinchilla Intuition

The Chinchilla result shifted practice toward training smaller models on more tokens than earlier Kaplan-style compute-optimal recommendations.

A common rule of thumb from that work is roughly tens of tokens per parameter, often quoted near 20 tokens per parameter for compute-optimal dense LMs.
## References

- OpenAI Scaling Laws for Neural Language Models: https://openai.com/index/scaling-laws-for-neural-language-models/
- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OpenAI InstructGPT / RLHF: https://openai.com/index/instruction-following/
- Ouyang et al., Training language models to follow instructions with human feedback: https://arxiv.org/abs/2203.02155
- Schulman et al., Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Rafailov et al., Direct Preference Optimization: https://arxiv.org/abs/2305.18290
