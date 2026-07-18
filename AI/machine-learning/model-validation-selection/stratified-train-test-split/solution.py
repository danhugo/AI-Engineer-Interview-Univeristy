"""Reference solutions for Stratified Train-Test Split."""

import numpy as np

def stratified_train_test_split_indices(y, test_size=0.25, seed=0):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train, test = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = rng.permutation(idx)
        n_test = int(np.ceil(len(idx) * test_size))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.array(sorted(train)), np.array(sorted(test))

def class_proportions(y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    return dict(zip(classes.tolist(), (counts / counts.sum()).tolist()))

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    y = np.array([0]*8 + [1]*2)
    train, test = stratified_train_test_split_indices(y, 0.25, seed=1)
    check(set(y[test]) == {0, 1}, "test contains both classes")
    print("PASS  test contains both classes")
    check(len(test) == 3 and len(train) == 7, "split sizes")
    print("PASS  split sizes")
    props = class_proportions(y[test])
    check(props[1] > 0, "minority represented")
    print("PASS  minority represented")
    print("All tests passed.")
