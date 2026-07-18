# Hierarchical Agglomerative Clustering - Interview Knowledge Sheet

## Intuition

Agglomerative clustering starts with every point as its own cluster and repeatedly merges the closest clusters.

This creates a hierarchy of merges, often visualized as a dendrogram.

---

## 1. Linkage

Linkage defines distance between clusters:

- single: closest pair of points
- complete: farthest pair of points
- average: average pairwise distance
- Ward: merge that minimally increases within-cluster variance

---

## 2. Practical Notes

Agglomerative clustering does not naturally predict labels for unseen points. It is mainly a clustering of the given dataset.
## References

- scikit-learn clustering user guide: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn cluster API: https://scikit-learn.org/stable/api/sklearn.cluster.html
- scikit-learn silhouette_score docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- scikit-learn Gaussian mixture user guide: https://scikit-learn.org/stable/modules/mixture.html
