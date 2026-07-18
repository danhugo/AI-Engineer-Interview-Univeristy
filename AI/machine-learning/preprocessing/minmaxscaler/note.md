# MinMaxScaler - Interview Knowledge Sheet

## Intuition

Min-max scaling maps each feature into a chosen range, often `[0, 1]`.

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}\cdot (b-a) + a
$$

where `[a, b]` is the target range.

---

## 1. When It Helps

It is useful when features need bounded ranges, such as image pixels or models sensitive to scale.

Unlike standard scaling, it does not center the feature around zero.

---

## 2. Outliers

Min-max scaling is sensitive to outliers because one extreme value sets the range.

Fit min and max on training data only to avoid leakage.
## References

- scikit-learn preprocessing API: https://scikit-learn.org/stable/api/sklearn.preprocessing.html
- scikit-learn OneHotEncoder docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- scikit-learn impute user guide: https://scikit-learn.org/stable/modules/impute.html
