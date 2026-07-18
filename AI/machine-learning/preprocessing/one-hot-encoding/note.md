# One-Hot Encoding - Interview Knowledge Sheet

## Intuition

One-hot encoding turns a categorical feature into binary indicator columns.

```text
color = red/blue/green -> color_red, color_blue, color_green
```

This lets models that expect numbers use categorical inputs without inventing an artificial order.

---

## 1. Fit Then Transform

Fit learns the categories from training data.

Transform uses those learned categories to create columns.

This matters because validation/test data must use the same columns as training.

---

## 2. Unknown Categories

Production data may contain categories not seen during training.

Common choices are to raise an error or encode unknown categories as all zeros for that feature.

---

## 3. Avoiding Leakage

Learn categories on the training split only. If categories are discovered from the full dataset, validation data influences preprocessing.
## References

- scikit-learn preprocessing API: https://scikit-learn.org/stable/api/sklearn.preprocessing.html
- scikit-learn OneHotEncoder docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- scikit-learn impute user guide: https://scikit-learn.org/stable/modules/impute.html
