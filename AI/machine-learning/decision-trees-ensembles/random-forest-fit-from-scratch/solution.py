"""Reference solutions for Random Forest Fit from Scratch."""

import numpy as np

def bootstrap_indices(n_samples, seed=None):
    return np.random.default_rng(seed).integers(0,n_samples,size=n_samples)

def gini(y):
    _,c=np.unique(y,return_counts=True); p=c/c.sum(); return float(1-np.sum(p**2))

def majority(y):
    cls,c=np.unique(y,return_counts=True); return cls[np.argmax(c)]

def fit_random_stump(X,y,features):
    parent=gini(y); best=(features[0],float(np.median(X[:,features[0]])),-np.inf)
    for j in features:
        vals=np.unique(X[:,j])
        for t in (vals[:-1]+vals[1:])/2:
            left=X[:,j]<=t
            if left.sum()==0 or left.sum()==len(y): continue
            gain=parent-left.mean()*gini(y[left])-(~left).mean()*gini(y[~left])
            if gain>best[2]: best=(j,float(t),float(gain))
    left=X[:,best[0]]<=best[1]
    return {"feature":best[0],"threshold":best[1],"left":majority(y[left]),"right":majority(y[~left])}

def predict_stump(stump,X):
    X=np.asarray(X,dtype=float); left=X[:,stump["feature"]]<=stump["threshold"]
    return np.where(left,stump["left"],stump["right"])

def fit_random_stump_forest(X,y,n_estimators=10,max_features=1,seed=0):
    X=np.asarray(X,dtype=float); y=np.asarray(y); rng=np.random.default_rng(seed); forest=[]
    for _ in range(n_estimators):
        rows=rng.integers(0,len(y),size=len(y)); feats=rng.choice(X.shape[1],size=max_features,replace=False)
        forest.append(fit_random_stump(X[rows],y[rows],feats))
    return forest

def predict_random_stump_forest(forest,X):
    votes=np.vstack([predict_stump(t,X) for t in forest]).T; out=[]
    for row in votes:
        cls,c=np.unique(row,return_counts=True); out.append(cls[np.argmax(c)])
    return np.asarray(out)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    idx=bootstrap_indices(5,seed=0); check(len(idx)==5 and idx.min()>=0 and idx.max()<5,"bootstrap indices"); print("PASS  bootstrap indices")
    X=np.array([[0,0],[0,1],[1,0],[1,1],[2,1],[2,2]],float); y=np.array([0,0,0,1,1,1])
    pred=predict_random_stump_forest(fit_random_stump_forest(X,y,7,1,4),X)
    check(pred.shape==y.shape and (pred==y).mean()>=.5,"random stump forest"); print("PASS  random stump forest")
    print("All tests passed.")
