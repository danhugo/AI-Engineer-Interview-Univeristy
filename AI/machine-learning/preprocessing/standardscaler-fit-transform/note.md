# StandardScaler Fit and Transform - Interview Knowledge Sheet

## Intuition

Standard scaling puts each feature on a comparable scale.

For each feature:

$$
z = \frac{x - \mu}{\sigma}
$$

The transformed training feature has mean near 0 and standard deviation near 1.

---

## 1. Why It Helps

Scaling matters for distance-based models, gradient-based optimization, SVMs, k-means, PCA, and regularized linear models.

It matters less for tree splits because trees compare thresholds feature-by-feature.

---

## 2. Leakage Rule

Fit `mu` and `sigma` on training data only.

Use the same stored values to transform validation and test data.
## References

- scikit-learn preprocessing API: https://scikit-learn.org/stable/api/sklearn.preprocessing.html
- scikit-learn OneHotEncoder docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- scikit-learn impute user guide: https://scikit-learn.org/stable/modules/impute.html
