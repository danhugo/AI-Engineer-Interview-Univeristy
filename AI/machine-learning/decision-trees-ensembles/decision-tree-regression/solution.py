"""Reference solutions for Decision Tree Regression."""

import numpy as np

def sse(y):
    y=np.asarray(y,dtype=float); return 0.0 if y.size==0 else float(np.sum((y-y.mean())**2))

def best_sse_stump(X,y):
    X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float); parent=sse(y); best=(None,None,-np.inf)
    for j in range(X.shape[1]):
        vals=np.unique(X[:,j])
        for t in (vals[:-1]+vals[1:])/2:
            left=X[:,j]<=t
            if left.sum()==0 or left.sum()==len(y): continue
            red=parent-sse(y[left])-sse(y[~left])
            if red>best[2]: best=(j,float(t),float(red))
    return best

def fit_regression_stump(X,y):
    X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float); f,t,r=best_sse_stump(X,y); left=X[:,f]<=t
    return {"feature":f,"threshold":t,"reduction":r,"left_value":float(y[left].mean()),"right_value":float(y[~left].mean())}

def predict_regression_stump(model,X):
    X=np.asarray(X,dtype=float); left=X[:,model["feature"]]<=model["threshold"]
    return np.where(left,model["left_value"],model["right_value"]).astype(float)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    check(np.isclose(sse([1,2,3]),2.0),"sse"); print("PASS  sse")
    X=np.array([[0.],[1.],[2.],[3.]]); y=np.array([1.,1.2,4.,4.2]); f,t,r=best_sse_stump(X,y)
    check(f==0 and np.isclose(t,1.5) and r>8,"best regression stump"); print("PASS  best regression stump")
    pred=predict_regression_stump(fit_regression_stump(X,y),[[.5],[2.5]])
    check(np.allclose(pred,[1.1,4.1]),"regression stump predictions"); print("PASS  regression stump predictions")
    print("All tests passed.")
