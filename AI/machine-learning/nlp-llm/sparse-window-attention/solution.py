"""Reference solutions for Sparse Window Attention."""


def make_window_mask(seq_len, window_size, causal=False):
    """Return a boolean mask showing which key positions each query can attend to."""
    import numpy as np

    positions = np.arange(seq_len)
    query_pos = positions[:, None]
    key_pos = positions[None, :]

    if causal:
        return (key_pos <= query_pos) & (key_pos >= query_pos - window_size)
    return np.abs(query_pos - key_pos) <= window_size


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    import numpy as np

    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def window_attention(q, k, v, window_size, causal=False):
    """Compute scaled dot-product attention with a local window mask."""
    import numpy as np

    dim = q.shape[-1]
    scores = q @ k.T / np.sqrt(dim)
    mask = make_window_mask(q.shape[0], window_size, causal=causal)
    masked_scores = np.where(mask, scores, -1e30)
    weights = softmax(masked_scores, axis=-1)
    return weights @ v


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
