"""Reference solutions for AdaBoost Fit Method."""

import numpy as np

def adaboost_alpha(weighted_error):
    e=np.clip(weighted_error,1e-12,1-1e-12); return float(.5*np.log((1-e)/e))

def update_sample_weights(weights,y_true,y_pred,alpha):
    w=np.asarray(weights,float); y=np.asarray(y_true,float); p=np.asarray(y_pred,float); new=w*np.exp(-alpha*y*p); return new/new.sum()

def weighted_vote(alphas,predictions):
    scores=np.asarray(predictions,float)@np.asarray(alphas,float); return np.where(scores>=0,1,-1)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    a=adaboost_alpha(.25); check(np.isclose(a,.5493061443340549),"alpha"); print("PASS  alpha")
    nw=update_sample_weights([.25,.25,.25,.25],[1,1,-1,-1],[1,-1,-1,-1],a); check(np.isclose(nw.sum(),1) and nw[1]>nw[0],"weight update"); print("PASS  weight update")
    votes=weighted_vote([.7,.2],[[1,-1],[-1,-1],[1,1]]); check(np.array_equal(votes,[1,-1,1]),"weighted vote"); print("PASS  weighted vote")
    print("All tests passed.")
