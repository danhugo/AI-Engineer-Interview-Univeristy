# RLAIF Reward Model - Interview Knowledge Sheet

## Intuition

RLAIF means reinforcement learning from AI feedback.

The reward model is trained from preferences labeled by another model, a constitution, or AI judge prompts instead of direct human labels.

The reward-model loss is usually still pairwise preference learning.

---

## 1. Pairwise Reward Model

Given prompt `x`, preferred answer `y_w`, and rejected answer `y_l`:

$$
P(y_w \succ y_l|x)=\sigma(r(x,y_w)-r(x,y_l))
$$

The loss is:

$$
-\log \sigma(r_w-r_l)
$$

---

## 2. Tradeoff

AI feedback can scale faster and cheaper than human feedback, but it can copy judge-model biases and miss things humans care about.

Good systems audit AI labels and mix them with human evaluation when quality matters.
## References

- OpenAI Scaling Laws for Neural Language Models: https://openai.com/index/scaling-laws-for-neural-language-models/
- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OpenAI InstructGPT / RLHF: https://openai.com/index/instruction-following/
- Ouyang et al., Training language models to follow instructions with human feedback: https://arxiv.org/abs/2203.02155
- Schulman et al., Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Rafailov et al., Direct Preference Optimization: https://arxiv.org/abs/2305.18290
