# Entropy-based Split Selection - Interview Knowledge Sheet

## Intuition

Entropy measures uncertainty in the class label.

If a node is pure, uncertainty is zero. If classes are evenly mixed, uncertainty is high.

$$
H = -\sum_k p_k \log_2(p_k)
$$

A good split reduces uncertainty.

---

## 1. Information Gain

$$
H_{split} = \frac{n_L}{n}H_L + \frac{n_R}{n}H_R
$$

$$
IG = H_{parent} - H_{split}
$$

The best threshold has the largest information gain.

---

## 2. Entropy vs Gini

Both reward pure children. Entropy has an information-theoretic meaning and is related to reducing log loss for leaf probability predictions. In practice, entropy and Gini often choose similar splits.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
