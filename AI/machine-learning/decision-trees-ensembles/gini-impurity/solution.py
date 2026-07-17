"""Reference solutions for Gini Impurity."""

import numpy as np

def class_probabilities(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.array([]), np.array([], dtype=float)
    classes, counts = np.unique(labels, return_counts=True)
    return classes, counts / counts.sum()

def gini_impurity(labels):
    _, probs = class_probabilities(labels)
    return float(1.0 - np.sum(probs ** 2)) if probs.size else 0.0

def weighted_gini_split(left_labels, right_labels):
    left = np.asarray(left_labels); right = np.asarray(right_labels)
    n = left.size + right.size
    return 0.0 if n == 0 else float((left.size/n)*gini_impurity(left) + (right.size/n)*gini_impurity(right))

def gini_gain(parent_labels, left_labels, right_labels):
    return float(gini_impurity(parent_labels) - weighted_gini_split(left_labels, right_labels))

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    y=np.array([0,0,1,1])
    check(np.isclose(gini_impurity(y),0.5),"balanced binary gini"); print("PASS  balanced binary gini")
    check(np.isclose(gini_impurity([1,1,1]),0.0),"pure gini"); print("PASS  pure gini")
    check(np.isclose(weighted_gini_split([0,0],[1,1]),0.0),"perfect split impurity"); print("PASS  perfect split impurity")
    check(np.isclose(gini_gain(y,[0,0],[1,1]),0.5),"gini gain"); print("PASS  gini gain")
    print("All tests passed.")
