"""
AI / Machine Learning — Feature Scaling
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""

import numpy as np


def fit_standard_scaler(X):
    """
    Return per-feature mean and safe standard deviation.

    HINT:
      Use population std with axis=0. Replace zero std values with 1.
    """
    pass


def transform_standard_scaler(X, mean, scale):
    """
    Standardize X using existing mean and scale.
    """
    pass


def fit_minmax_scaler(X):
    """
    Return per-feature min and safe range.

    HINT:
      Replace zero ranges with 1.
    """
    pass


def transform_minmax_scaler(X, data_min, data_range):
    """
    Scale X to [0, 1] using existing min and range.
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_standard_scaler():
    X_train = np.array([[1.0, 10.0, 5.0],
                        [3.0, 14.0, 5.0],
                        [5.0, 16.0, 5.0]])
    mean, scale = fit_standard_scaler(X_train)
    check(np.allclose(mean, [3.0, 40.0 / 3.0, 5.0]), f"mean wrong: {mean}")
    check(np.allclose(scale, [np.sqrt(8.0 / 3.0), np.sqrt(56.0 / 9.0), 1.0]),
          f"scale wrong: {scale}")

    X_scaled = transform_standard_scaler(X_train, mean, scale)
    check(np.allclose(X_scaled.mean(axis=0), [0.0, 0.0, 0.0]), "scaled mean wrong")
    check(np.allclose(X_scaled[:, 2], [0.0, 0.0, 0.0]), "constant column should become zero")
    print("PASS  standard_scaler")


def test_minmax_scaler():
    X_train = np.array([[1.0, 10.0, 5.0],
                        [3.0, 14.0, 5.0],
                        [5.0, 16.0, 5.0]])
    data_min, data_range = fit_minmax_scaler(X_train)
    check(np.allclose(data_min, [1.0, 10.0, 5.0]), f"min wrong: {data_min}")
    check(np.allclose(data_range, [4.0, 6.0, 1.0]), f"range wrong: {data_range}")

    X_scaled = transform_minmax_scaler(X_train, data_min, data_range)
    expected = np.array([[0.0, 0.0, 0.0],
                         [0.5, 2.0 / 3.0, 0.0],
                         [1.0, 1.0, 0.0]])
    check(np.allclose(X_scaled, expected), f"minmax scaled wrong: {X_scaled}")
    print("PASS  minmax_scaler")


def test_use_train_stats_for_test_data():
    X_train = np.array([[0.0], [10.0]])
    X_test = np.array([[20.0]])
    mean, scale = fit_standard_scaler(X_train)
    scaled_test = transform_standard_scaler(X_test, mean, scale)
    check(np.allclose(scaled_test, [[3.0]]), f"must use train stats, got: {scaled_test}")
    print("PASS  train_stats_for_test_data")


if __name__ == "__main__":
    test_standard_scaler()
    test_minmax_scaler()
    test_use_train_stats_for_test_data()
    print("All tests passed.")
