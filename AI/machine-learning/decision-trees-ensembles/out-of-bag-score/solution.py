"""Reference solutions for Out-of-Bag Score."""

import numpy as np

def oob_indices(n_samples, bootstrap_idx):
    used=np.zeros(n_samples,dtype=bool); used[np.asarray(bootstrap_idx,dtype=int)]=True; return np.flatnonzero(~used)

def oob_accuracy(y_true, tree_predictions, bootstrap_indices_per_tree):
    y_true=np.asarray(y_true); correct=counted=0
    boot_sets=[set(map(int,b)) for b in bootstrap_indices_per_tree]
    for i,y in enumerate(y_true):
        votes=[np.asarray(pred)[i] for pred,boot in zip(tree_predictions,boot_sets) if i not in boot]
        if not votes: continue
        cls,c=np.unique(votes,return_counts=True); correct+=int(cls[np.argmax(c)]==y); counted+=1
    return correct/counted if counted else np.nan

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    check(np.array_equal(oob_indices(5,[0,0,2,4,4]),[1,3]),"oob indices"); print("PASS  oob indices")
    y=np.array([0,1,1,0]); preds=[np.array([0,1,0,0]),np.array([1,1,1,0]),np.array([0,0,1,1])]; boots=[np.array([0,0,1,1]),np.array([1,1,2,2]),np.array([2,2,3,3])]
    check(np.isclose(oob_accuracy(y,preds,boots),.5),"oob accuracy"); print("PASS  oob accuracy")
    print("All tests passed.")
