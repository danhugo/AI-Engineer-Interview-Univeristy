"""Reference solutions for BIRCH Clustering."""

import numpy as np

def clustering_feature(points):
    X=np.asarray(points,float)
    return {"n": len(X), "linear_sum": X.sum(axis=0), "squared_sum": float(np.sum(X*X))}

def cf_centroid(cf):
    return cf["linear_sum"] / cf["n"]

def merge_clustering_features(a, b):
    return {"n": a["n"]+b["n"], "linear_sum": a["linear_sum"]+b["linear_sum"], "squared_sum": a["squared_sum"]+b["squared_sum"]}

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    cf=clustering_feature([[1,2],[3,4]]); check(cf["n"]==2 and np.allclose(cf["linear_sum"],[4,6]),"cf summary"); print("PASS  cf summary")
    check(np.allclose(cf_centroid(cf),[2,3]),"centroid"); print("PASS  centroid")
    m=merge_clustering_features(cf, clustering_feature([[5,6]])); check(m["n"]==3 and np.allclose(cf_centroid(m),[3,4]),"merge"); print("PASS  merge")
    print("All tests passed.")
