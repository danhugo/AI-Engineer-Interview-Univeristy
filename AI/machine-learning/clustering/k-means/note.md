# K-Means - Interview Knowledge Sheet

## Intuition

K-means groups points around `k` centroids.

It repeats two steps:

```text
assign each point to nearest centroid -> move each centroid to mean of assigned points
```

---

## 1. Objective

K-means minimizes within-cluster sum of squares, also called inertia:

$$
\sum_{i=1}^{n}\|x_i - \mu_{c_i}\|^2
$$

Each update step does not increase this objective, but the algorithm can converge to a local optimum.

---

## 2. Practical Notes

K-means assumes roughly spherical, similarly sized clusters under Euclidean distance. It is sensitive to feature scale, outliers, initialization, and the chosen `k`.
## References

- scikit-learn clustering user guide: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn cluster API: https://scikit-learn.org/stable/api/sklearn.cluster.html
- scikit-learn silhouette_score docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- scikit-learn Gaussian mixture user guide: https://scikit-learn.org/stable/modules/mixture.html
