"""
AI / Machine Learning - Rotary Positional Encoding RoPE
PRACTICE FILE
"""


def rotate_half(x):
    """
    For pairs [a, b], return [-b, a].

    x shape: (..., dim), where dim is even.
    """
    pass


def rope_cos_sin(seq_len, dim, base=10000.0):
    """
    Return (cos, sin), each with shape (seq_len, dim), for RoPE.
    """
    pass


def apply_rope(x, cos, sin):
    """
    Apply RoPE to x.

    x shape: (..., seq_len, dim)
    cos/sin shape: (seq_len, dim)
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def run_tests():
    import numpy as np

    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    check(np.array_equal(rotate_half(x), np.array([[-2.0, 1.0, -4.0, 3.0]])), "rotate_half")
    print("PASS  rotate_half")

    cos, sin = rope_cos_sin(seq_len=3, dim=4)
    check(cos.shape == (3, 4) and sin.shape == (3, 4), "cos/sin shape")
    check(np.allclose(cos[0], 1.0) and np.allclose(sin[0], 0.0), "position zero")
    print("PASS  angle tables")

    tokens = np.arange(12, dtype=float).reshape(1, 3, 4)
    rotated = apply_rope(tokens, cos, sin)
    check(np.allclose(rotated[:, 0, :], tokens[:, 0, :]), "position 0 unchanged")
    check(np.allclose(np.linalg.norm(rotated, axis=-1), np.linalg.norm(tokens, axis=-1)), "norm preserved")
    print("PASS  apply_rope")


if __name__ == "__main__":
    run_tests()
    print("All tests passed.")
