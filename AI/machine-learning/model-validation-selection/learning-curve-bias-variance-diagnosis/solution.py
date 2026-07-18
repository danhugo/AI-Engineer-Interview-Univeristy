"""Reference solutions for Learning Curve Generator for Bias-Variance Diagnosis."""

import numpy as np

def diagnose_learning_curve(train_scores, valid_scores, gap_threshold=0.1, low_score_threshold=0.7):
    train_mean = float(np.mean(train_scores[-1]))
    valid_mean = float(np.mean(valid_scores[-1]))
    gap = train_mean - valid_mean
    if train_mean < low_score_threshold and valid_mean < low_score_threshold:
        return "high_bias"
    if gap > gap_threshold:
        return "high_variance"
    return "reasonable_fit"

def summarize_learning_curve(train_sizes, train_scores, valid_scores):
    return {
        "train_sizes": list(train_sizes),
        "mean_train_scores": np.mean(train_scores, axis=1).tolist(),
        "mean_valid_scores": np.mean(valid_scores, axis=1).tolist(),
    }

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    diag = diagnose_learning_curve([[0.6, 0.62]], [[0.58, 0.6]])
    check(diag == "high_bias", "bias diagnosis")
    print("PASS  bias diagnosis")
    diag = diagnose_learning_curve([[0.98, 0.97]], [[0.72, 0.74]])
    check(diag == "high_variance", "variance diagnosis")
    print("PASS  variance diagnosis")
    s = summarize_learning_curve([10, 20], [[1, .9], [.8, .8]], [[.7, .6], [.75, .75]])
    check(s["mean_train_scores"] == [0.95, 0.8], "summary means")
    print("PASS  summary means")
    print("All tests passed.")
