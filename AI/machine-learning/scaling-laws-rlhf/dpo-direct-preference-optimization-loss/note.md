# DPO Direct Preference Optimization Loss - Interview Knowledge Sheet

## Intuition

DPO trains directly on preference pairs without first training a separate reward model and then running PPO.

For a prompt, preferred answer `y_w`, rejected answer `y_l`, policy `pi_theta`, and reference `pi_ref`, DPO increases the winner's log-probability ratio relative to the loser.

---

## 1. Loss

$$
L_{DPO} = -\log \sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)
$$

The term inside the sigmoid is the preference margin.

---

## 2. Intuition

If the policy improves the winner relative to the reference and/or downweights the loser relative to the reference, the margin grows and loss falls.

`beta` controls how strongly deviations from the reference translate into preference margins.
## References

- OpenAI Scaling Laws for Neural Language Models: https://openai.com/index/scaling-laws-for-neural-language-models/
- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OpenAI InstructGPT / RLHF: https://openai.com/index/instruction-following/
- Ouyang et al., Training language models to follow instructions with human feedback: https://arxiv.org/abs/2203.02155
- Schulman et al., Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Rafailov et al., Direct Preference Optimization: https://arxiv.org/abs/2305.18290
