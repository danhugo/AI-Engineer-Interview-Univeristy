# Combined RM and LLM Quality Filter - Interview Knowledge Sheet

## Intuition

A reward model score is useful but not complete.

A practical quality filter often combines:

- reward model preference score
- LLM-as-judge quality score
- rule checks such as length, toxicity, format, or refusal policy

The goal is not to prove an answer is perfect. It is to reject obvious low-quality or risky candidates before training or serving.

---

## 1. Combined Score

A simple weighted score is:

$$
S = w_{RM}s_{RM} + w_{judge}s_{judge} - w_{risk}s_{risk}
$$

Then apply thresholds.

---

## 2. Why Combine Signals

Reward models can be gamed. LLM judges can be inconsistent. Rules can be brittle.

Combining signals reduces dependence on any single weak evaluator.
## References

- OpenAI Scaling Laws for Neural Language Models: https://openai.com/index/scaling-laws-for-neural-language-models/
- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OpenAI InstructGPT / RLHF: https://openai.com/index/instruction-following/
- Ouyang et al., Training language models to follow instructions with human feedback: https://arxiv.org/abs/2203.02155
- Schulman et al., Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Rafailov et al., Direct Preference Optimization: https://arxiv.org/abs/2305.18290
