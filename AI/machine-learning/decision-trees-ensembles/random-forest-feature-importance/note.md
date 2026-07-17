# Random Forest Feature Importance - Interview Knowledge Sheet

## Intuition

Feature importance asks which features the trained model relied on most.

Tree ensembles often use impurity decrease importance: a feature gets credit whenever a split on that feature reduces impurity.

---

## 1. Mean Decrease in Impurity

For one split:

$$
\text{weighted decrease} = \frac{n_t}{N}\left(I_t - \frac{n_L}{n_t}I_L - \frac{n_R}{n_t}I_R\right)
$$

Sum this over splits using the feature, then normalize so importances sum to 1. In a forest, aggregate across trees.

---

## 2. Caveats

Impurity importance can favor high-cardinality continuous features. Permutation importance is a useful check: shuffle one feature and measure validation score drop. Importance is model reliance, not causal proof.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
