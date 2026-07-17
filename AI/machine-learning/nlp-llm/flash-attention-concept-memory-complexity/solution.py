"""Reference solutions for FlashAttention Concept and Memory Complexity."""


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    import numpy as np

    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def standard_attention(q, k, v):
    """Compute ordinary scaled dot-product attention for 2D NumPy arrays."""
    import numpy as np

    scores = q @ k.T / np.sqrt(q.shape[-1])
    return softmax(scores, axis=-1) @ v


def online_softmax_update(old_max, old_sum, new_scores):
    """Update running softmax max and denominator with a new 1D score block."""
    import numpy as np

    block_max = np.max(new_scores)
    new_max = max(old_max, block_max)
    rescaled_old = np.exp(old_max - new_max) * old_sum if old_sum != 0 else 0.0
    block_sum = np.exp(new_scores - new_max).sum()
    return new_max, rescaled_old + block_sum


def tiled_attention(q, k, v, block_size):
    """Compute exact attention by streaming K/V in blocks with online softmax."""
    import numpy as np

    scale = np.sqrt(q.shape[-1])
    outputs = []
    for query in q:
        running_max = -np.inf
        running_sum = 0.0
        running_num = np.zeros(v.shape[-1])

        for start in range(0, k.shape[0], block_size):
            kb = k[start:start + block_size]
            vb = v[start:start + block_size]
            scores = query @ kb.T / scale
            block_max = np.max(scores)
            new_max = max(running_max, block_max)

            old_scale = np.exp(running_max - new_max) if running_sum != 0 else 0.0
            exp_scores = np.exp(scores - new_max)

            running_num = running_num * old_scale + exp_scores @ vb
            running_sum = running_sum * old_scale + exp_scores.sum()
            running_max = new_max

        outputs.append(running_num / running_sum)

    return np.stack(outputs, axis=0)


def attention_matrix_elements(seq_len, heads=1, batch=1):
    """Return the number of elements in a dense attention matrix."""
    return batch * heads * seq_len * seq_len


def qkv_output_elements(seq_len, head_dim, heads=1, batch=1):
    """Return element count for Q, K, V, and output tensors."""
    return 4 * batch * heads * seq_len * head_dim


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
