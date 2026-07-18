# KL-Penalized Reward Shaping RLHF - Interview Knowledge Sheet

## Intuition

RLHF should improve helpfulness without letting the policy drift too far from the reference model.

The common shaped reward is:

$$
r_{total}(x,y)=r_{RM}(x,y)-\beta KL(\pi_\theta(\cdot|x)\|\pi_{ref}(\cdot|x))
$$

For sampled tokens, the KL term is often estimated from log-probability differences:

$$
\log \pi_\theta(y_t|x,y_{<t}) - \log \pi_{ref}(y_t|x,y_{<t})
$$

---

## 1. Why Penalize KL

The reward model is imperfect. If optimization only maximizes reward, the policy can exploit reward model weaknesses.

KL penalty keeps outputs close to the SFT/reference model unless reward improvement is worth the drift.

---

## 2. Beta

Large `beta` means stronger anchoring to the reference model.

Small `beta` allows more movement but increases reward hacking and distribution-shift risk.
## References

- OpenAI Scaling Laws for Neural Language Models: https://openai.com/index/scaling-laws-for-neural-language-models/
- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OpenAI InstructGPT / RLHF: https://openai.com/index/instruction-following/
- Ouyang et al., Training language models to follow instructions with human feedback: https://arxiv.org/abs/2203.02155
- Schulman et al., Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Rafailov et al., Direct Preference Optimization: https://arxiv.org/abs/2305.18290
