"""
AI / Machine Learning — Gradient Clipping
PRACTICE FILE
"""

import numpy as np


def np_total_l2_norm(arrays):
    """
    Return the global L2 norm across a list of gradient arrays.

    HINT:
      Sum squared entries across all arrays, then take sqrt.
    """
    pass


def np_clip_grad_norm(arrays, max_norm, eps=1e-12):
    """
    Return clipped copies of arrays and the original total norm.

    HINT:
      If total_norm > max_norm, multiply every array by max_norm / (total_norm + eps).
    """
    pass


def np_clip_grad_value(arrays, clip_value):
    """
    Return copies with each entry clamped to [-clip_value, clip_value].
    """
    pass


def torch_clip_grad_norm_example(max_norm=5.0):
    """
    Create a tensor with a large gradient, clip it, and return (before, after).

    HINT:
      Use torch.nn.utils.clip_grad_norm_([param], max_norm).
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_numpy_norm_clipping():
    grads = [np.array([3.0, 4.0]), np.array([0.0])]
    total = np_total_l2_norm(grads)
    check(abs(total - 5.0) < 1e-8, f"total norm wrong: {total}")
    print("PASS  np_total_l2_norm")

    clipped, original_norm = np_clip_grad_norm(grads, max_norm=2.5)
    check(abs(original_norm - 5.0) < 1e-8, f"original norm wrong: {original_norm}")
    check(np.allclose(clipped[0], [1.5, 2.0]), f"clipped grad wrong: {clipped[0]}")
    check(abs(np_total_l2_norm(clipped) - 2.5) < 1e-8, "clipped norm should equal max_norm")
    print("PASS  np_clip_grad_norm")

    unchanged, _ = np_clip_grad_norm(grads, max_norm=10.0)
    check(np.allclose(unchanged[0], grads[0]), "small gradient should stay unchanged")
    print("PASS  no clipping when below max_norm")


def test_numpy_value_clipping():
    grads = [np.array([-3.0, -0.5, 2.0])]
    clipped = np_clip_grad_value(grads, clip_value=1.0)
    check(np.allclose(clipped[0], [-1.0, -0.5, 1.0]), f"value clipping wrong: {clipped[0]}")
    print("PASS  np_clip_grad_value")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return

    before, after = torch_clip_grad_norm_example(max_norm=5.0)
    check(abs(before - 10.0) < 1e-6, f"torch norm before wrong: {before}")
    check(abs(after - 5.0) < 1e-5, f"torch norm after wrong: {after}")
    print("PASS  torch_clip_grad_norm_example")


if __name__ == "__main__":
    test_numpy_norm_clipping()
    test_numpy_value_clipping()
    test_torch()
    print("All tests passed.")
