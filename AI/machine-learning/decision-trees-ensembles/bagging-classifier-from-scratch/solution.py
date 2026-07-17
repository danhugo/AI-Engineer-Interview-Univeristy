"""Reference solutions for Bagging Classifier from Scratch."""

import numpy as np

def majority(y):
    cls,c=np.unique(y,return_counts=True); return cls[np.argmax(c)]

def fit_bagging_majority_classifiers(y,n_estimators=5,seed=0):
    y=np.asarray(y); rng=np.random.default_rng(seed); models=[]
    for _ in range(n_estimators):
        idx=rng.integers(0,len(y),size=len(y)); models.append({"class":majority(y[idx]),"bootstrap_idx":idx})
    return models

def predict_bagging_majority_classifiers(models,n_samples):
    votes=np.array([[m["class"]]*n_samples for m in models]).T; out=[]
    for row in votes:
        cls,c=np.unique(row,return_counts=True); out.append(cls[np.argmax(c)])
    return np.asarray(out)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    models=fit_bagging_majority_classifiers(np.array([0,0,0,1,1]),9,1); check(len(models)==9,"model count"); print("PASS  model count")
    pred=predict_bagging_majority_classifiers(models,3); check(pred.shape==(3,) and set(pred).issubset({0,1}),"bagging predictions"); print("PASS  bagging predictions")
    print("All tests passed.")
