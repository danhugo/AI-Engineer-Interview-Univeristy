"""Reference solutions for K-Means."""

import numpy as np

def assign_to_centroids(X, centroids):
    X=np.asarray(X,float); C=np.asarray(centroids,float)
    d=((X[:,None,:]-C[None,:,:])**2).sum(axis=2)
    return np.argmin(d,axis=1)

def recompute_centroids(X, labels, k):
    X=np.asarray(X,float); labels=np.asarray(labels)
    return np.vstack([X[labels==j].mean(axis=0) for j in range(k)])

def inertia(X, labels, centroids):
    X=np.asarray(X,float); C=np.asarray(centroids,float); labels=np.asarray(labels)
    return float(np.sum((X-C[labels])**2))

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    X=np.array([[0,0],[0,1],[5,5],[6,5]],float); C=np.array([[0,0],[5,5]],float)
    lab=assign_to_centroids(X,C); check(np.array_equal(lab,[0,0,1,1]),"assignments"); print("PASS  assignments")
    new=recompute_centroids(X,lab,2); check(np.allclose(new,[[0,.5],[5.5,5]]),"centroids"); print("PASS  centroids")
    check(np.isclose(inertia(X,lab,new),1.0),"inertia"); print("PASS  inertia")
    print("All tests passed.")
