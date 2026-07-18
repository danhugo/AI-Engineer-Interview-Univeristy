# Reward Model Calibration - Interview Knowledge Sheet

## Intuition

A reward model is calibrated when its preference probabilities match observed preference frequencies.

If it says 80% confidence, the preferred answer should win about 80% of the time among similar examples.

---

## 1. Pairwise Probability

For winner reward `r_w` and loser reward `r_l`:

$$
P(w \succ l)=\sigma(r_w-r_l)
$$

Calibration checks whether those predicted probabilities match empirical outcomes.

---

## 2. Expected Calibration Error

Bucket predictions by confidence. For each bucket:

$$
ECE = \sum_b \frac{n_b}{n}|acc(b)-conf(b)|
$$

For reward models, `acc` is how often the preferred label wins inside the bucket.

---

## 3. Why It Matters

An uncalibrated reward model can make PPO or filtering too aggressive. Calibration helps compare thresholds and detect overconfident reward scores.
## References

- OpenAI Scaling Laws for Neural Language Models: https://openai.com/index/scaling-laws-for-neural-language-models/
- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- OpenAI InstructGPT / RLHF: https://openai.com/index/instruction-following/
- Ouyang et al., Training language models to follow instructions with human feedback: https://arxiv.org/abs/2203.02155
- Schulman et al., Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Rafailov et al., Direct Preference Optimization: https://arxiv.org/abs/2305.18290
