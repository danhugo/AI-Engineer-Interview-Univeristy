# Sigmoidal Accuracy-to-Log-Likelihood Scaling Law Fit - Interview Knowledge Sheet

## Intuition

Cross-entropy or log-likelihood can improve smoothly while task accuracy stays flat, then jumps, then saturates.

That shape is often sigmoidal:

```text
low accuracy plateau -> transition region -> high accuracy plateau
```

A simple fit is:

$$
A(\ell)=A_{min}+\frac{A_{max}-A_{min}}{1+\exp(k(\ell-\ell_0))}
$$

where lower loss `ell` usually means higher accuracy.

---

## 1. Why It Happens

Accuracy is thresholded. A model gets no extra credit until the correct answer crosses the decision boundary.

Log-likelihood sees confidence improvements before accuracy changes.

---

## 2. Interview Use

Use this to explain why validation loss can be a smoother scaling signal than benchmark accuracy, especially near saturation or chance-level performance.
## References

- OpenAI Scaling Laws for Neural Language Models: https://openai.com/index/scaling-laws-for-neural-language-models/
- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OpenAI InstructGPT / RLHF: https://openai.com/index/instruction-following/
- Ouyang et al., Training language models to follow instructions with human feedback: https://arxiv.org/abs/2203.02155
- Schulman et al., Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Rafailov et al., Direct Preference Optimization: https://arxiv.org/abs/2305.18290
