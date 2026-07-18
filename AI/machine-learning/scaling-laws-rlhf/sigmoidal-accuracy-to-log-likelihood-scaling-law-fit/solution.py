"""Reference solutions for Sigmoidal Accuracy-to-Log-Likelihood Scaling Law Fit."""

import numpy as np

def sigmoid_accuracy(loss, min_acc, max_acc, midpoint, slope):
    loss = np.asarray(loss, dtype=float)
    return min_acc + (max_acc - min_acc) / (1.0 + np.exp(slope * (loss - midpoint)))

def accuracy_slope_at_midpoint(min_acc, max_acc, slope):
    return float(-(max_acc - min_acc) * slope / 4.0)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    acc = sigmoid_accuracy([2.0, 1.0, 0.0], 0.2, 0.9, 1.0, 3.0)
    check(acc[0] < acc[1] < acc[2], "accuracy increases as loss falls")
    print("PASS  monotonic direction")
    check(np.isclose(sigmoid_accuracy(1.0, 0.2, 0.9, 1.0, 3.0), 0.55), "midpoint")
    print("PASS  midpoint")
    check(accuracy_slope_at_midpoint(0.2, 0.9, 3.0) < 0, "slope sign")
    print("PASS  slope sign")
    print("All tests passed.")
