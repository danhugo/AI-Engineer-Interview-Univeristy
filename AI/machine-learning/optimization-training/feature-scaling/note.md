# Feature Scaling — Interview Knowledge Sheet

## Intuition

Feature scaling puts input columns on comparable numeric ranges.

Without scaling, a feature measured in thousands can dominate a feature measured in decimals even if it is not more important:

```
income: 85000
age: 42
```

Many optimization algorithms and distance-based models behave better when features have similar scale.

---

## 1. Standardization

Standardization subtracts the training mean and divides by the training standard deviation:

$$
z = \frac{x - \mu}{\sigma}
$$

After standardization, each feature usually has:

- mean near `0`
- standard deviation near `1`

Use the statistics learned on training data for validation and test data.

---

## 2. Min-Max Scaling

Min-max scaling maps a feature to a chosen range, commonly `[0, 1]`:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

For a custom range `[a, b]`:

$$
x_{\text{scaled}} = x'(b-a) + a
$$

It preserves the order of values, but it is sensitive to outliers.

---

## 3. Why Scaling Helps Optimization

Gradient descent can zig-zag when features have very different scales.

For linear models, large-scale features can produce larger gradient components, making one learning rate hard to choose for all weights.

Scaling makes the loss surface more balanced, so gradient descent often converges faster.

---

## 4. Models That Care

Feature scaling is usually important for:

- gradient descent linear and logistic regression
- neural networks
- k-nearest neighbors
- k-means
- SVMs, especially with RBF kernels
- PCA
- regularized linear models

It is usually less important for:

- decision trees
- random forests
- gradient-boosted trees

Tree splits depend mostly on feature ordering, not raw distance or gradient scale.

---

## 5. Data Leakage

Fit scalers only on training data:

```python
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std
```

Do not compute scaling statistics on the full dataset before splitting. That leaks test-set information into training.

---

## 6. Edge Cases

Constant columns have zero standard deviation or zero range.

Practical scalers avoid division by zero by leaving those columns as zeros after centering or by using a scale of `1`.

Outliers can badly affect mean/std and min/max. Robust scaling uses median and interquartile range when outliers are a major concern.

---

## 7. Interview Gotchas

- Fit the scaler on training data only.
- Apply the same stored transformation to validation/test data.
- Standardization is not the same as min-max scaling.
- Scaling is important for distance, gradient, kernel, and regularized methods.
- Scaling usually does not change tree model splits.
- Handle zero-variance features explicitly.

---

## References

- scikit-learn `StandardScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- scikit-learn preprocessing user guide: https://scikit-learn.org/stable/modules/preprocessing.html
- scikit-learn common pitfalls, data leakage: https://scikit-learn.org/stable/common_pitfalls.html
