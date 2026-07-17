"""
AI / Machine Learning - Multi-Head Attention From Scratch
PRACTICE FILE
"""


def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    """
    pass


def split_heads(x, num_heads):
    """
    Convert x from (batch, seq_len, d_model) to (batch, num_heads, seq_len, head_dim).
    """
    pass


def combine_heads(x):
    """
    Convert x from (batch, num_heads, seq_len, head_dim) to (batch, seq_len, d_model).
    """
    pass


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute scaled dot-product attention.

    q, k, v shape: (batch, heads, seq_len, head_dim)
    mask shape: broadcastable to (batch, heads, seq_len, seq_len), True means allowed.
    Return (output, weights).
    """
    pass


def multi_head_attention(x, wq, wk, wv, wo, num_heads, mask=None):
    """
    Project x into q/k/v, run attention, combine heads, and apply output projection.
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def run_tests():
    import numpy as np

    x = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    heads = split_heads(x, num_heads=2)
    check(heads.shape == (2, 2, 3, 2), "split shape")
    check(np.array_equal(combine_heads(heads), x), "combine should invert split")
    print("PASS  split/combine heads")

    q = split_heads(np.eye(4)[None, :, :], num_heads=2)
    out, weights = scaled_dot_product_attention(q, q, q)
    check(out.shape == q.shape, "attention output shape")
    check(np.allclose(weights.sum(axis=-1), 1.0), "weights sum to 1")
    print("PASS  scaled attention")

    d_model = 4
    ident = np.eye(d_model)
    y = multi_head_attention(x, ident, ident, ident, ident, num_heads=2)
    check(y.shape == x.shape, "multi-head output shape")
    print("PASS  multi-head attention")


if __name__ == "__main__":
    run_tests()
    print("All tests passed.")
