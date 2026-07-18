# Label Encoding - Interview Knowledge Sheet

## Intuition

Label encoding maps class labels to integer IDs.

```text
cat -> 0, dog -> 1, rabbit -> 2
```

This is usually appropriate for target labels `y`, not nominal input features.

---

## 1. Why For Targets

Classifiers often need labels represented as integers internally.

The integer ID is just an identifier, not a numeric distance.

---

## 2. Feature Caveat

Using label encoding on input categories can mislead models that treat numbers as ordered.

For nominal input features, use one-hot encoding or another categorical method.
## References

- scikit-learn preprocessing API: https://scikit-learn.org/stable/api/sklearn.preprocessing.html
- scikit-learn OneHotEncoder docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- scikit-learn impute user guide: https://scikit-learn.org/stable/modules/impute.html
