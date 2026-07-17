"""Reference solutions for LightGBM Objective Function."""

import numpy as np

def leaf_score(grad_sum,hess_sum,l2):
    return float((grad_sum**2)/(hess_sum+l2))

def split_gain(g_left,h_left,g_right,h_right,l2=0.0,min_gain_to_split=0.0):
    parent=leaf_score(g_left+g_right,h_left+h_right,l2); gain=leaf_score(g_left,h_left,l2)+leaf_score(g_right,h_right,l2)-parent
    return float(gain-min_gain_to_split)

def histogram_gradient_sums(feature_bins,gradients,hessians,n_bins):
    bins=np.asarray(feature_bins,int); g=np.asarray(gradients,float); h=np.asarray(hessians,float); gs=np.zeros(n_bins); hs=np.zeros(n_bins)
    for b,gi,hi in zip(bins,g,h): gs[b]+=gi; hs[b]+=hi
    return gs,hs

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    check(np.isclose(leaf_score(3,2,1),3),"leaf score"); print("PASS  leaf score")
    check(np.isclose(split_gain(-2,1,2,1,l2=1),4.0),"split gain"); print("PASS  split gain")
    gs,hs=histogram_gradient_sums([0,1,1,2],[1,2,-1,3],[1,1,2,1.5],3); check(np.allclose(gs,[1,1,3]) and np.allclose(hs,[1,3,1.5]),"histogram sums"); print("PASS  histogram sums")
    print("All tests passed.")
