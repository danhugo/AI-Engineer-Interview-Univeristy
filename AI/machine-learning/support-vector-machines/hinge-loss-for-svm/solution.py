"""Reference solutions for Hinge Loss for SVM."""

import numpy as np

def hinge_loss(y_true, scores):
    return np.maximum(0.0, 1.0 - np.asarray(y_true, dtype=float) * np.asarray(scores, dtype=float))

def mean_hinge_loss(y_true, scores):
    return float(np.mean(hinge_loss(y_true, scores)))

def svm_primal_objective(w, y_true, scores, C):
    return float(0.5 * np.dot(w, w) + C * np.sum(hinge_loss(y_true, scores)))

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    losses = hinge_loss([1,-1,1], [2, .5, .2])
    check(np.allclose(losses, [0, 1.5, .8]), "hinge losses")
    print("PASS  hinge losses")
    check(np.isclose(mean_hinge_loss([1,1], [1,0]), .5), "mean loss")
    print("PASS  mean loss")
    obj = svm_primal_objective(np.array([1.,1.]), [1], [0.], C=2)
    check(np.isclose(obj, 3.0), "objective")
    print("PASS  objective")
    print("All tests passed.")
