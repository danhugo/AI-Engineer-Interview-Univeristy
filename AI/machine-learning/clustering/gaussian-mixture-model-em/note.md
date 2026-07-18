# Gaussian Mixture Model with EM - Interview Knowledge Sheet

## Intuition

A Gaussian mixture model assumes data comes from a weighted mixture of Gaussian distributions.

Each point has a soft assignment to each component, not a hard cluster label.

---

## 1. EM Algorithm

Expectation step:

$$
r_{ik} = P(z_i=k \mid x_i)
$$

Maximization step updates weights, means, and covariances using those responsibilities.

For means:

$$
\mu_k = \frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}}
$$

---

## 2. K-Means Connection

GMM generalizes K-means by modeling covariance and soft membership. K-means is like a hard-assignment, equal-spherical-variance special case.
## References

- scikit-learn clustering user guide: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn cluster API: https://scikit-learn.org/stable/api/sklearn.cluster.html
- scikit-learn silhouette_score docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- scikit-learn Gaussian mixture user guide: https://scikit-learn.org/stable/modules/mixture.html
