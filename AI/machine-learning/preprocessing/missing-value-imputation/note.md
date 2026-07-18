# Missing Value Imputation - Interview Knowledge Sheet

## Intuition

Imputation fills missing values using statistics learned from the training data.

Common simple strategies:

- mean for numeric features
- median for skewed numeric features or outlier robustness
- mode for categorical features

---

## 1. Formula

Mean imputation for a numeric feature:

$$
x_i = \begin{cases}
x_i & \text{if observed}\\
\bar{x}_{train} & \text{if missing}
\end{cases}
$$

---

## 2. Leakage Rule

Compute imputation values from training data only.

Validation/test missing values should be filled with the training statistic, not their own statistic.

---

## 3. Missingness Can Be Signal

Sometimes the fact that a value is missing is predictive.

Add a missing indicator when missingness itself may carry information.
## References

- scikit-learn preprocessing API: https://scikit-learn.org/stable/api/sklearn.preprocessing.html
- scikit-learn OneHotEncoder docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- scikit-learn impute user guide: https://scikit-learn.org/stable/modules/impute.html
