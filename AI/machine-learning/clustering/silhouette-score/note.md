# Silhouette Score - Interview Knowledge Sheet

## Intuition

Silhouette score measures whether points are closer to their own cluster than to the nearest other cluster.

For one point:

- `a`: mean distance to points in the same cluster
- `b`: mean distance to points in the nearest other cluster

$$
s = \frac{b-a}{\max(a,b)}
$$

---

## 1. Interpretation

Scores near 1 mean well-separated clusters.

Scores near 0 mean overlapping clusters.

Negative scores suggest points may be assigned to the wrong cluster.

---

## 2. Caveat

Silhouette often favors compact convex clusters and may not fairly judge density-based shapes such as DBSCAN clusters.
## References

- scikit-learn clustering user guide: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn cluster API: https://scikit-learn.org/stable/api/sklearn.cluster.html
- scikit-learn silhouette_score docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- scikit-learn Gaussian mixture user guide: https://scikit-learn.org/stable/modules/mixture.html
