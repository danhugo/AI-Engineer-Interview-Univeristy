# Stratified Train-Test Split - Interview Knowledge Sheet

## Intuition

A random split can accidentally distort class balance, especially for small or imbalanced datasets.

Stratification splits each class separately, then combines the pieces.

The goal is:

$$
P_{train}(y=k) \approx P_{test}(y=k) \approx P(y=k)
$$

---

## 1. Why It Matters

If a rare class is 2% of the dataset, a naive split might put too few rare examples in validation.

Then metrics become noisy or misleading.

Stratification makes the split preserve label proportions as much as integer counts allow.

---

## 2. What It Does Not Solve

Stratification does not prevent leakage from duplicate users, time ordering, or grouped samples.

If examples from the same user appear in train and test, use group-aware splitting instead.
## References

- scikit-learn model selection user guide: https://scikit-learn.org/stable/model_selection.html
- scikit-learn learning curve docs: https://scikit-learn.org/stable/modules/learning_curve.html
