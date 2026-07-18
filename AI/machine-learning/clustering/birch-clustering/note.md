# BIRCH Clustering - Interview Knowledge Sheet

## Intuition

BIRCH compresses a large dataset into many small summary subclusters, then optionally clusters those summaries.

The summary is called a clustering feature.

---

## 1. Clustering Feature

For a subcluster:

- `N`: number of points
- `LS`: linear sum of points
- `SS`: sum of squared norms

The centroid is:

$$
\mu = \frac{LS}{N}
$$

This lets BIRCH summarize data without storing every point in memory.

---

## 2. When It Helps

BIRCH is useful for large datasets when you want incremental compression before final clustering. The threshold controls how tight subclusters must be.
## References

- scikit-learn clustering user guide: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn cluster API: https://scikit-learn.org/stable/api/sklearn.cluster.html
- scikit-learn silhouette_score docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- scikit-learn Gaussian mixture user guide: https://scikit-learn.org/stable/modules/mixture.html
