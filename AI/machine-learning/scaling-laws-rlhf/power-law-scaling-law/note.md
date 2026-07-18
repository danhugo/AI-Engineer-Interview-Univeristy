# Power-Law Scaling Law - Interview Knowledge Sheet

## Intuition

Scaling laws say model loss often improves predictably as you increase scale.

The important shape is a power law: every multiplicative increase in scale gives a smaller but still predictable improvement.

A common simplified form is:

$$
L(x) = L_\infty + A x^{-\alpha}
$$

where `x` can be parameters, data tokens, or compute.

---

## 1. Log-Log View

Subtract the irreducible floor:

$$
L(x)-L_\infty = A x^{-\alpha}
$$

Taking logs gives a line:

$$
\log(L-L_\infty)=\log A - \alpha \log x
$$

So the slope on a log-log plot is `-alpha`.

---

## 2. What It Means

Scaling laws are empirical fits, not guarantees. They are useful for forecasting loss and choosing training budgets, but they do not directly predict every downstream capability or safety property.
## References

- OpenAI Scaling Laws for Neural Language Models: https://openai.com/index/scaling-laws-for-neural-language-models/
- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OpenAI InstructGPT / RLHF: https://openai.com/index/instruction-following/
- Ouyang et al., Training language models to follow instructions with human feedback: https://arxiv.org/abs/2203.02155
- Schulman et al., Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Rafailov et al., Direct Preference Optimization: https://arxiv.org/abs/2305.18290
