"""Reference solutions for DBSCAN."""

import numpy as np

def radius_neighbors(X, point_index, eps):
    X=np.asarray(X,float); d=np.linalg.norm(X-X[point_index], axis=1)
    return np.flatnonzero(d <= eps)

def core_points(X, eps, min_samples):
    return np.array([len(radius_neighbors(X,i,eps)) >= min_samples for i in range(len(X))])

def simple_dbscan_labels(X, eps, min_samples):
    X=np.asarray(X,float); core=core_points(X,eps,min_samples); labels=np.full(len(X),-1,int); cluster_id=0
    for i in range(len(X)):
        if labels[i] != -1 or not core[i]: continue
        stack=[i]; labels[i]=cluster_id
        while stack:
            j=stack.pop()
            for n in radius_neighbors(X,j,eps):
                if labels[n] == -1:
                    labels[n]=cluster_id
                    if core[n]: stack.append(n)
        cluster_id += 1
    return labels

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    X=np.array([[0,0],[0,.1],[5,5],[5.1,5],[10,10]],float)
    core=core_points(X,.2,2); check(np.array_equal(core,[1,1,1,1,0]),"core points"); print("PASS  core points")
    lab=simple_dbscan_labels(X,.2,2); check(np.array_equal(lab,[0,0,1,1,-1]),"labels"); print("PASS  labels")
    print("All tests passed.")
