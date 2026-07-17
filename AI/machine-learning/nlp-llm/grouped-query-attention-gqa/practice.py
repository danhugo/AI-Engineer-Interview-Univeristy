"""
AI / Machine Learning - Grouped Query Attention GQA
PRACTICE FILE
"""


def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    """
    pass


def repeat_kv_heads(x, num_query_heads):
    """
    Repeat KV heads so they align with query heads.

    x shape: (batch, kv_heads, seq_len, head_dim)
    Return shape: (batch, num_query_heads, seq_len, head_dim)
    """
    pass


def grouped_query_attention(q, k, v, mask=None):
    """
    Compute grouped-query attention.

    q shape: (batch, query_heads, seq_len, head_dim)
    k/v shape: (batch, kv_heads, seq_len, head_dim)
    """
    pass


def kv_cache_elements(batch, seq_len, layers, kv_heads, head_dim):
    """
    Return number of scalar elements in a K+V cache.
    """
    pass


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
