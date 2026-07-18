# K-Means++ Initialization - Interview Knowledge Sheet

## Intuition

K-means is sensitive to starting centroids.

K-means++ chooses spread-out initial centers so the algorithm starts with better coverage of the data.

---

## 1. Procedure

1. Pick the first center randomly.
2. For each point, compute distance squared to its nearest chosen center.
3. Pick the next center with probability proportional to that squared distance.
4. Repeat until `k` centers are chosen.

Points far from existing centers are more likely to become new centers.

---

## 2. Why It Helps

Better initialization usually lowers final inertia and reduces the need for many random restarts.
## References

- scikit-learn clustering user guide: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn cluster API: https://scikit-learn.org/stable/api/sklearn.cluster.html
- scikit-learn silhouette_score docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- scikit-learn Gaussian mixture user guide: https://scikit-learn.org/stable/modules/mixture.html
