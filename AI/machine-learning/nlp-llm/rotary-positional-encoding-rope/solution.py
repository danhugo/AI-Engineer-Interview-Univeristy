"""Reference solutions for Rotary Positional Encoding RoPE."""


def rotate_half(x):
    """For pairs [a, b], return [-b, a]."""
    import numpy as np

    if x.shape[-1] % 2 != 0:
        raise ValueError("last dimension must be even")
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    rotated = np.stack([-pairs[..., 1], pairs[..., 0]], axis=-1)
    return rotated.reshape(x.shape)


def rope_cos_sin(seq_len, dim, base=10000.0):
    """Return cosine and sine tables for RoPE."""
    import numpy as np

    if dim % 2 != 0:
        raise ValueError("dim must be even")
    pair_indices = np.arange(0, dim, 2)
    inv_freq = 1.0 / (base ** (pair_indices / dim))
    positions = np.arange(seq_len)[:, None]
    angles = positions * inv_freq[None, :]
    angles = np.repeat(angles, repeats=2, axis=-1)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin):
    """Apply RoPE to x."""
    return x * cos + rotate_half(x) * sin


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
