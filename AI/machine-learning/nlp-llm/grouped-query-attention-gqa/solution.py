"""Reference solutions for Grouped Query Attention GQA."""


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    import numpy as np

    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def repeat_kv_heads(x, num_query_heads):
    """Repeat KV heads so they align with query heads."""
    import numpy as np

    batch, kv_heads, seq_len, head_dim = x.shape
    if num_query_heads % kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by kv_heads")
    repeats = num_query_heads // kv_heads
    return np.repeat(x, repeats=repeats, axis=1).reshape(batch, num_query_heads, seq_len, head_dim)


def grouped_query_attention(q, k, v, mask=None):
    """Compute grouped-query attention."""
    import numpy as np

    k_repeated = repeat_kv_heads(k, q.shape[1])
    v_repeated = repeat_kv_heads(v, q.shape[1])
    scores = q @ np.swapaxes(k_repeated, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e30)
    weights = softmax(scores, axis=-1)
    return weights @ v_repeated, weights


def kv_cache_elements(batch, seq_len, layers, kv_heads, head_dim):
    """Return number of scalar elements in a K+V cache."""
    return 2 * batch * seq_len * layers * kv_heads * head_dim


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def run_tests():
    import numpy as np

    kv = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
    repeated = repeat_kv_heads(kv, num_query_heads=4)
    check(repeated.shape == (1, 4, 3, 4), "repeated shape")
    check(np.array_equal(repeated[:, 0], kv[:, 0]) and np.array_equal(repeated[:, 1], kv[:, 0]), "first group")
    check(np.array_equal(repeated[:, 2], kv[:, 1]) and np.array_equal(repeated[:, 3], kv[:, 1]), "second group")
    print("PASS  repeat KV heads")

    q = np.ones((1, 4, 3, 2))
    k = np.ones((1, 2, 3, 2))
    v = np.arange(1 * 2 * 3 * 2, dtype=float).reshape(1, 2, 3, 2)
    out, weights = grouped_query_attention(q, k, v)
    check(out.shape == q.shape, "GQA output shape")
    check(weights.shape == (1, 4, 3, 3), "GQA weights shape")
    check(np.allclose(weights.sum(axis=-1), 1.0), "weights sum to 1")
    print("PASS  grouped-query attention")

    mha_cache = kv_cache_elements(batch=2, seq_len=100, layers=12, kv_heads=8, head_dim=64)
    gqa_cache = kv_cache_elements(batch=2, seq_len=100, layers=12, kv_heads=2, head_dim=64)
    check(mha_cache == 4 * gqa_cache, "KV cache reduction")
    print("PASS  cache sizing")


if __name__ == "__main__":
    run_tests()
    print("All tests passed.")
