"""
AI / Machine Learning - FlashAttention Concept and Memory Complexity
PRACTICE FILE
"""


def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    """
    pass


def standard_attention(q, k, v):
    """
    Compute ordinary scaled dot-product attention for 2D NumPy arrays.
    """
    pass


def online_softmax_update(old_max, old_sum, new_scores):
    """
    Update running softmax max and denominator with a new 1D score block.

    Return (new_max, new_sum).
    """
    pass


def tiled_attention(q, k, v, block_size):
    """
    Compute exact attention one query row at a time, streaming K/V in blocks.

    This is a tiny educational version of the online-softmax idea, not a GPU kernel.
    """
    pass


def attention_matrix_elements(seq_len, heads=1, batch=1):
    """
    Return the number of elements in a dense attention matrix.
    """
    pass


def qkv_output_elements(seq_len, head_dim, heads=1, batch=1):
    """
    Return element count for Q, K, V, and output tensors.
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def run_tests():
    import numpy as np

    scores = np.array([1.0, 2.0, -3.0, 0.5])
    m, l = -np.inf, 0.0
    m, l = online_softmax_update(m, l, scores[:2])
    m, l = online_softmax_update(m, l, scores[2:])
    expected_l = np.exp(scores - scores.max()).sum()
    check(np.isclose(m, scores.max()) and np.isclose(l, expected_l), "online softmax stats")
    print("PASS  online softmax update")

    rng = np.random.default_rng(3)
    q = rng.normal(size=(5, 4))
    k = rng.normal(size=(5, 4))
    v = rng.normal(size=(5, 4))
    ref = standard_attention(q, k, v)
    tiled = tiled_attention(q, k, v, block_size=2)
    check(np.allclose(ref, tiled, atol=1e-10), "tiled attention should match standard")
    print("PASS  tiled exact attention")

    dense = attention_matrix_elements(seq_len=1024, heads=8, batch=2)
    qkvo = qkv_output_elements(seq_len=1024, head_dim=64, heads=8, batch=2)
    check(dense == 2 * 8 * 1024 * 1024, "dense attention element count")
    check(qkvo == 4 * 2 * 8 * 1024 * 64, "QKV/output element count")
    check(dense > qkvo, "attention matrix dominates at long sequence")
    print("PASS  memory sizing")


if __name__ == "__main__":
    run_tests()
    print("All tests passed.")
