"""Reference solutions for Reward Model Calibration."""

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))

def reward_preference_probability(reward_a, reward_b):
    return sigmoid(np.asarray(reward_a) - np.asarray(reward_b))

def expected_calibration_error(probabilities, outcomes, n_bins=10):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    p = reward_preference_probability([2.0,0.0], [0.0,0.0])
    check(p[0] > p[1], "probability order")
    print("PASS  probability order")
    ece = expected_calibration_error([0.8,0.8,0.2,0.2], [1,1,0,0], n_bins=2)
    check(np.isclose(ece, 0.2), "ece")
    print("PASS  ece")
    print("All tests passed.")
