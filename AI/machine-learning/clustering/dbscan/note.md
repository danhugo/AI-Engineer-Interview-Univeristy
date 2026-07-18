# DBSCAN - Interview Knowledge Sheet

## Intuition

DBSCAN finds dense regions separated by sparse regions.

It does not require choosing the number of clusters upfront.

---

## 1. Core Points

A point is a core point if at least `min_samples` points, including itself in scikit-learn convention, are within distance `eps`.

Clusters grow by connecting density-reachable core points. Points that are not reachable from any core point are labeled noise.

---

## 2. Parameters

Smaller `eps` or larger `min_samples` means a stricter density requirement.

DBSCAN can find non-convex clusters, but it struggles when clusters have very different densities.
## References

- scikit-learn clustering user guide: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn cluster API: https://scikit-learn.org/stable/api/sklearn.cluster.html
- scikit-learn silhouette_score docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- scikit-learn Gaussian mixture user guide: https://scikit-learn.org/stable/modules/mixture.html
