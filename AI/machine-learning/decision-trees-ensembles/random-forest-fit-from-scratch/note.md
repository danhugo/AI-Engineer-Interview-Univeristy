# Random Forest Fit from Scratch - Interview Knowledge Sheet

## Intuition

A random forest is many noisy decision trees averaged together. Each tree is trained on a bootstrap sample of rows, and each split usually considers a random subset of features.

This creates diverse trees. Averaging diverse trees reduces variance.

---

## 1. Bagging

Bootstrap aggregation means sample `n` rows with replacement, train one tree, repeat many times, then average probabilities or take majority vote.

$$
\hat{y} = \operatorname{mode}(T_1(x), T_2(x), \ldots, T_B(x))
$$

---

## 2. Feature Randomness

At each split, random forests consider only `m` of `p` features. This decorrelates trees and prevents one dominant feature from controlling every tree. If all trees make the same error, averaging cannot help.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
