"""Reference solutions for K-Means++ Initialization."""

import numpy as np

def nearest_center_squared_distances(X, centers):
    X=np.asarray(X,float); C=np.asarray(centers,float)
    return ((X[:,None,:]-C[None,:,:])**2).sum(axis=2).min(axis=1)

def kmeans_plus_plus_probabilities(X, centers):
    d2=nearest_center_squared_distances(X, centers)
    total=d2.sum()
    return np.full(len(d2),1/len(d2)) if total==0 else d2/total

def choose_next_center_index(X, centers, seed=0):
    probs=kmeans_plus_plus_probabilities(X, centers)
    return int(np.random.default_rng(seed).choice(len(probs), p=probs))

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    X=np.array([[0,0],[1,0],[10,0]],float)
    d2=nearest_center_squared_distances(X, [[0,0]])
    check(np.allclose(d2,[0,1,100]),"nearest distances"); print("PASS  nearest distances")
    p=kmeans_plus_plus_probabilities(X, [[0,0]])
    check(np.isclose(p.sum(),1) and p[2]>p[1]>p[0],"probabilities"); print("PASS  probabilities")
    print("All tests passed.")
