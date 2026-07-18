"""Reference solutions for Silhouette Score."""

import numpy as np

def silhouette_sample_scores(X, labels):
    X=np.asarray(X,float); labels=np.asarray(labels); scores=[]
    D=np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(axis=2))
    for i in range(len(X)):
        same=(labels==labels[i]); same[i]=False
        a=D[i,same].mean() if same.any() else 0.0
        b=min(D[i,labels==lab].mean() for lab in np.unique(labels) if lab != labels[i])
        scores.append((b-a)/max(a,b) if max(a,b)>0 else 0.0)
    return np.array(scores)

def silhouette_score(X, labels):
    return float(silhouette_sample_scores(X, labels).mean())

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    X=np.array([[0,0],[0,1],[5,5],[5,6]],float); labels=np.array([0,0,1,1])
    s=silhouette_sample_scores(X,labels); check(np.all(s>0.7),"sample scores"); print("PASS  sample scores")
    check(silhouette_score(X,labels)>0.7,"mean score"); print("PASS  mean score")
    print("All tests passed.")
