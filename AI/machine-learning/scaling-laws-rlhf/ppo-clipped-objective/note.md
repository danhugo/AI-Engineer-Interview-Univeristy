# PPO Clipped Objective - Interview Knowledge Sheet

## Intuition

PPO updates a policy but tries to prevent updates from moving too far in one step.

It compares the new policy probability to the old policy probability:

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

Then it clips that ratio inside a trust-region-like range.

---

## 1. Clipped Objective

$$
L^{CLIP}(\theta)=E\left[\min(r_t A_t, \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right]
$$

If an update would improve the objective only by making the ratio too extreme, clipping limits the incentive.

---

## 2. RLHF Use

In classic RLHF, PPO fine-tunes the policy against reward-model scores plus a KL penalty to the reference model.
## References

- OpenAI Scaling Laws for Neural Language Models: https://openai.com/index/scaling-laws-for-neural-language-models/
- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OpenAI InstructGPT / RLHF: https://openai.com/index/instruction-following/
- Ouyang et al., Training language models to follow instructions with human feedback: https://arxiv.org/abs/2203.02155
- Schulman et al., Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Rafailov et al., Direct Preference Optimization: https://arxiv.org/abs/2305.18290
