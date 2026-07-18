"""Reference solutions for Train/Val/Test Split with No Leakage."""

import numpy as np

def train_val_test_split_indices(n_samples, val_size=0.2, test_size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_test = int(round(n_samples * test_size))
    n_val = int(round(n_samples * val_size))
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return np.sort(train), np.sort(val), np.sort(test)

def check_no_overlap(*splits):
    sets = [set(map(int, s)) for s in splits]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                return False
    return True

def fit_train_mean_transform(train_values, other_values):
    mean = float(np.mean(train_values))
    return np.asarray(other_values, dtype=float) - mean, mean

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    tr, va, te = train_val_test_split_indices(10, 0.2, 0.2, seed=0)
    check(len(tr) == 6 and len(va) == 2 and len(te) == 2, "split sizes")
    print("PASS  split sizes")
    check(check_no_overlap(tr, va, te), "no overlap")
    print("PASS  no overlap")
    transformed, mean = fit_train_mean_transform([1, 2, 3], [10, 11])
    check(np.isclose(mean, 2.0) and np.allclose(transformed, [8, 9]), "train-only mean")
    print("PASS  train-only mean")
    print("All tests passed.")
