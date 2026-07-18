# Elbow Method for K Selection - Interview Knowledge Sheet

## Intuition

K-means inertia always decreases as `k` increases.

The elbow method looks for the point where adding more clusters gives much smaller improvement.

```text
large gain -> large gain -> small gain -> small gain
                         elbow
```

---

## 1. What It Plots

For each `k`, fit K-means and record inertia:

$$
\sum_i \|x_i - \mu_{c_i}\|^2
$$

Choose a `k` near the bend, not necessarily the absolute minimum.

---

## 2. Caveat

The elbow can be ambiguous. Use domain knowledge, silhouette score, stability checks, or downstream usefulness to support the choice.
## References

- scikit-learn clustering user guide: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn cluster API: https://scikit-learn.org/stable/api/sklearn.cluster.html
- scikit-learn silhouette_score docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- scikit-learn Gaussian mixture user guide: https://scikit-learn.org/stable/modules/mixture.html
