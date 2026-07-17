"""Reference solutions for Entropy-based Split Selection."""

import numpy as np

def entropy(labels, base=2):
    labels=np.asarray(labels)
    if labels.size==0: return 0.0
    _, counts=np.unique(labels, return_counts=True)
    p=counts/counts.sum()
    return float(-np.sum(p*(np.log(p)/np.log(base))))

def information_gain(parent_labels, left_labels, right_labels):
    left=np.asarray(left_labels); right=np.asarray(right_labels); n=left.size+right.size
    if n==0: return 0.0
    return float(entropy(parent_labels)-((left.size/n)*entropy(left)+(right.size/n)*entropy(right)))

def best_threshold_by_information_gain(feature, labels):
    x=np.asarray(feature,dtype=float); y=np.asarray(labels); order=np.argsort(x); xs=x[order]; ys=y[order]
    best_t=None; best_gain=-np.inf
    for i in range(1,len(xs)):
        if xs[i]==xs[i-1]: continue
        t=(xs[i]+xs[i-1])/2; gain=information_gain(ys,ys[:i],ys[i:])
        if gain>best_gain: best_t=float(t); best_gain=float(gain)
    return best_t,best_gain

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    check(np.isclose(entropy([0,0,1,1]),1.0),"balanced entropy"); print("PASS  balanced entropy")
    check(np.isclose(entropy([1,1,1]),0.0),"pure entropy"); print("PASS  pure entropy")
    check(np.isclose(information_gain([0,0,1,1],[0,0],[1,1]),1.0),"perfect information gain"); print("PASS  perfect information gain")
    t,g=best_threshold_by_information_gain([0.1,0.2,0.8,0.9],[0,0,1,1])
    check(np.isclose(t,0.5) and np.isclose(g,1.0),"best threshold"); print("PASS  best threshold")
    print("All tests passed.")
