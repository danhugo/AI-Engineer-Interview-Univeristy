# LightGBM Objective Function - Interview Knowledge Sheet

## Intuition

LightGBM is gradient-boosted decision trees using gradient and Hessian statistics to choose leaf values and split gains.

The practical difference is not the high-level loss; it is how LightGBM builds trees efficiently.

---

## 1. Leaf-wise Growth

LightGBM grows trees leaf-wise: it repeatedly splits the leaf with the largest gain. This can reduce loss quickly but can overfit if uncontrolled. Important controls include `num_leaves`, `max_depth`, `min_data_in_leaf`, `min_gain_to_split`, `lambda_l1`, and `lambda_l2`.

---

## 2. Split Gain

With L2 regularization, the score intuition is:

$$
Score(I) = \frac{G_I^2}{H_I + \lambda}
$$

$$
Gain = Score(L) + Score(R) - Score(P) - \gamma
$$

---

## 3. Histogram Splits

LightGBM bins continuous features into histograms, accumulates gradient/Hessian sums per bin, and evaluates bin boundaries. This reduces memory and speeds split search.
## References

- LightGBM Features: https://lightgbm.readthedocs.io/en/stable/Features.html
- LightGBM Parameters: https://lightgbm.readthedocs.io/en/stable/Parameters.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
