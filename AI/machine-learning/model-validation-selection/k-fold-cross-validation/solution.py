"""Reference solutions for K-Fold Cross-Validation."""

import numpy as np

def k_fold_indices(n_samples, k, shuffle=False, seed=0):
    indices = np.arange(n_samples)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        valid = indices[start:start + size]
        train = np.setdiff1d(indices, valid, assume_unique=False)
        folds.append((train, valid))
        start += size
    return folds

def mean_cv_score(scores):
    scores = np.asarray(scores, dtype=float)
    return float(scores.mean()), float(scores.std(ddof=0))

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    folds = k_fold_indices(10, 3)
    check([len(v) for _, v in folds] == [4, 3, 3], "fold sizes")
    print("PASS  fold sizes")
    seen = np.sort(np.concatenate([v for _, v in folds]))
    check(np.array_equal(seen, np.arange(10)), "each sample validated once")
    print("PASS  each sample validated once")
    mean, std = mean_cv_score([0.8, 0.9, 1.0])
    check(np.isclose(mean, 0.9) and std > 0, "score summary")
    print("PASS  score summary")
    print("All tests passed.")
