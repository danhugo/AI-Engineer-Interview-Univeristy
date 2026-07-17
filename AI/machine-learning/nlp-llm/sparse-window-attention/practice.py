"""
AI / Machine Learning - Sparse Window Attention
PRACTICE FILE
"""


def make_window_mask(seq_len, window_size, causal=False):
    """
    Return a boolean NumPy array of shape (seq_len, seq_len).

    mask[i, j] should be True when query position i may attend to key position j.

    Bidirectional window:
      abs(i - j) <= window_size

    Causal window:
      i - window_size <= j <= i
    """
    pass


def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    """
    pass


def window_attention(q, k, v, window_size, causal=False):
    """
    Compute scaled dot-product attention with a local window mask.

    q, k, v are NumPy arrays with shape (seq_len, dim).
    Return an array with shape (seq_len, dim).
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def run_tests():
    import numpy as np

    mask = make_window_mask(seq_len=5, window_size=1, causal=False)
    expected = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=bool)
    check(np.array_equal(mask, expected), "bidirectional window mask")
    print("PASS  bidirectional mask")

    causal = make_window_mask(seq_len=5, window_size=2, causal=True)
    check(causal[3].tolist() == [False, True, True, True, False], "causal row 3")
    check(not causal[1, 2], "causal mask should block future keys")
    print("PASS  causal mask")

    q = np.eye(4)
    k = np.eye(4)
    v = np.arange(16, dtype=float).reshape(4, 4)
    out = window_attention(q, k, v, window_size=0, causal=False)
    check(np.allclose(out, v), "window size 0 should attend only to self")
    print("PASS  self-only attention")

    out_causal = window_attention(q, k, v, window_size=1, causal=True)
    check(np.allclose(out_causal[0], v[0]), "first causal token sees only itself")
    check(out_causal.shape == (4, 4), "output shape")
    print("PASS  window attention")


if __name__ == "__main__":
    run_tests()
    print("All tests passed.")
