"""
AI / Machine Learning - Early Stopping with Validation Loss
PRACTICE FILE
"""

import numpy as np


def best_epoch_and_value(values, mode="min", min_delta=0.0):
    """
    Return (best_epoch_index, best_value) using min_delta.

    HINT:
      For mode="min", a new value improves if value < best - min_delta.
      For mode="max", a new value improves if value > best + min_delta.
    """
    pass


def epochs_since_best(values, mode="min", min_delta=0.0):
    """
    Return how many epochs have passed since the last qualifying improvement.

    HINT:
      Use best_epoch_and_value, then subtract best_epoch from len(values) - 1.
    """
    pass


def should_stop_early(values, patience, mode="min", min_delta=0.0, start_from_epoch=0):
    """
    Return True if training should stop.

    HINT:
      Ignore early epochs before start_from_epoch.
      Stop when epochs_since_best >= patience.
    """
    pass


def restore_best_weight(weights_by_epoch, values, mode="min", min_delta=0.0):
    """
    Return a copy of the weight array from the best epoch.

    HINT:
      Find the best epoch, then return np.array(weights_by_epoch[best_epoch], copy=True).
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


VAL_LOSS = np.array([0.90, 0.72, 0.70, 0.705, 0.71, 0.69, 0.695, 0.697])
VAL_ACC = np.array([0.60, 0.66, 0.67, 0.675, 0.674])


def test_best_epoch():
    epoch, value = best_epoch_and_value(VAL_LOSS, mode="min")
    check(epoch == 5, f"best loss epoch wrong: {epoch}")
    check(abs(value - 0.69) < 1e-12, f"best loss value wrong: {value}")

    epoch, value = best_epoch_and_value(VAL_ACC, mode="max", min_delta=0.01)
    check(epoch == 3, f"best accuracy epoch with min_delta wrong: {epoch}")
    check(abs(value - 0.675) < 1e-12, f"best accuracy value wrong: {value}")
    print("PASS  best_epoch_and_value")


def test_stop_rule():
    check(epochs_since_best(VAL_LOSS, mode="min") == 2, "epochs_since_best wrong")
    check(should_stop_early(VAL_LOSS, patience=2, mode="min"), "should stop after patience")
    check(not should_stop_early(VAL_LOSS, patience=3, mode="min"), "should not stop yet")
    check(not should_stop_early(VAL_LOSS[:2], patience=1, start_from_epoch=3),
          "warm-up epochs should not stop")
    print("PASS  stop rule")


def test_restore_best_weight():
    weights = [np.array([epoch, epoch + 0.5]) for epoch in range(len(VAL_LOSS))]
    restored = restore_best_weight(weights, VAL_LOSS)
    check(np.allclose(restored, [5.0, 5.5]), f"restored wrong weight: {restored}")
    restored[0] = -999.0
    check(weights[5][0] == 5.0, "restored weight should be a copy")
    print("PASS  restore_best_weight")


if __name__ == "__main__":
    test_best_epoch()
    test_stop_rule()
    test_restore_best_weight()
    print("All tests passed.")
