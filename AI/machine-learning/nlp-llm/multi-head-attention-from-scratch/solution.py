"""Reference solutions for Multi-Head Attention From Scratch."""


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    import numpy as np

    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def split_heads(x, num_heads):
    """Convert (batch, seq_len, d_model) to (batch, heads, seq_len, head_dim)."""
    batch, seq_len, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    head_dim = d_model // num_heads
    return x.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)


def combine_heads(x):
    """Convert (batch, heads, seq_len, head_dim) to (batch, seq_len, d_model)."""
    batch, num_heads, seq_len, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, num_heads * head_dim)


def scaled_dot_product_attention(q, k, v, mask=None):
    """Compute scaled dot-product attention."""
    import numpy as np

    head_dim = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(head_dim)
    if mask is not None:
        scores = np.where(mask, scores, -1e30)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def multi_head_attention(x, wq, wk, wv, wo, num_heads, mask=None):
    """Project x into q/k/v, run attention, combine heads, and apply output projection."""
    q = split_heads(x @ wq, num_heads)
    k = split_heads(x @ wk, num_heads)
    v = split_heads(x @ wv, num_heads)
    attended, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    return combine_heads(attended) @ wo


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
