"""Reference solutions for Decision Tree Classification."""

import numpy as np

def gini(y):
    _, c=np.unique(y,return_counts=True); p=c/c.sum(); return float(1-np.sum(p**2))

def majority(y):
    cls,c=np.unique(y,return_counts=True); return cls[np.argmax(c)]

def best_gini_stump(X,y):
    X=np.asarray(X,dtype=float); y=np.asarray(y); parent=gini(y); best=(None,None,-np.inf)
    for j in range(X.shape[1]):
        vals=np.unique(X[:,j])
        for t in (vals[:-1]+vals[1:])/2:
            left=X[:,j]<=t
            if left.sum()==0 or left.sum()==len(y): continue
            gain=parent-left.mean()*gini(y[left])-(~left).mean()*gini(y[~left])
            if gain>best[2]: best=(j,float(t),float(gain))
    return best

def fit_decision_stump_classifier(X,y):
    X=np.asarray(X,dtype=float); y=np.asarray(y); f,t,g=best_gini_stump(X,y); left=X[:,f]<=t
    return {"feature":f,"threshold":t,"gain":g,"left_class":majority(y[left]),"right_class":majority(y[~left])}

def predict_decision_stump(model,X):
    X=np.asarray(X,dtype=float); left=X[:,model["feature"]]<=model["threshold"]
    return np.where(left,model["left_class"],model["right_class"])

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    X=np.array([[0.],[1.],[2.],[3.]]); y=np.array([0,0,1,1]); f,t,g=best_gini_stump(X,y)
    check(f==0 and np.isclose(t,1.5) and np.isclose(g,0.5),"best stump"); print("PASS  best stump")
    pred=predict_decision_stump(fit_decision_stump_classifier(X,y),[[.2],[2.5]])
    check(np.array_equal(pred,[0,1]),"stump predictions"); print("PASS  stump predictions")
    print("All tests passed.")
