"""Reference solutions for Hierarchical Agglomerative Clustering."""

import numpy as np

def pairwise_distances(X):
    X=np.asarray(X,float)
    return np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(axis=2))

def cluster_distance(X, cluster_a, cluster_b, linkage="single"):
    D=pairwise_distances(X); vals=D[np.ix_(cluster_a, cluster_b)]
    if linkage == "single": return float(vals.min())
    if linkage == "complete": return float(vals.max())
    if linkage == "average": return float(vals.mean())
    raise ValueError("unsupported linkage")

def closest_clusters(X, clusters, linkage="single"):
    best=None
    for i in range(len(clusters)):
        for j in range(i+1,len(clusters)):
            d=cluster_distance(X,clusters[i],clusters[j],linkage)
            if best is None or d < best[2]: best=(i,j,d)
    return best

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    X=np.array([[0,0],[0,1],[5,5]],float); clusters=[[0],[1],[2]]
    i,j,d=closest_clusters(X,clusters); check((i,j)==(0,1) and np.isclose(d,1),"closest clusters"); print("PASS  closest clusters")
    check(np.isclose(cluster_distance(X,[0,1],[2],"complete"), np.sqrt(50)),"complete linkage"); print("PASS  complete linkage")
    print("All tests passed.")
