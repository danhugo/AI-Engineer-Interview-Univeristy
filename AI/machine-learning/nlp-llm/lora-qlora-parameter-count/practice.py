"""AI / Machine Learning - LoRA / QLoRA Parameter Count Practice."""


def dense_linear_params(in_features, out_features, bias=False):
    """Return parameter count for a dense linear layer."""
    pass


def lora_linear_params(in_features, out_features, rank, bias=False):
    """Return trainable LoRA parameter count for one linear layer."""
    pass


def total_lora_params(layers):
    """Return total LoRA params for layer dicts with in_features/out_features/rank."""
    pass


def trainable_fraction(trainable_params, total_base_params):
    """Return trainable params divided by frozen base params."""
    pass


def qlora_memory_bytes(base_params, lora_params, base_bits=4, adapter_bytes=2,
                       adam_state_bytes=4, include_adam=True):
    """Return a rough QLoRA parameter and optimizer memory estimate."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def test_single_layer_counts():
    base = dense_linear_params(4096, 4096)
    lora = lora_linear_params(4096, 4096, rank=8)
    check(base == 16_777_216, f"base params wrong: {base}")
    check(lora == 65_536, f"LoRA params wrong: {lora}")
    check(close(trainable_fraction(lora, base), 1 / 256), "trainable fraction wrong")
    print("PASS  single layer counts")


def test_total_lora_params():
    layers = [
        {"name": "q_proj", "in_features": 4096, "out_features": 4096, "rank": 8},
        {"name": "v_proj", "in_features": 4096, "out_features": 4096, "rank": 8},
        {"name": "up_proj", "in_features": 4096, "out_features": 11008, "rank": 4},
    ]
    total = total_lora_params(layers)
    expected = 65_536 + 65_536 + 60_416
    check(total == expected, f"total LoRA params wrong: {total}")
    print("PASS  total LoRA params")


def test_qlora_memory():
    estimate = qlora_memory_bytes(
        base_params=1_000_000_000,
        lora_params=10_000_000,
        base_bits=4,
        adapter_bytes=2,
        adam_state_bytes=4,
        include_adam=True,
    )
    check(estimate["base_bytes"] == 500_000_000, f"base bytes wrong: {estimate}")
    check(estimate["adapter_bytes"] == 20_000_000, f"adapter bytes wrong: {estimate}")
    check(estimate["adam_bytes"] == 80_000_000, f"Adam bytes wrong: {estimate}")
    check(estimate["total_bytes"] == 600_000_000, f"total bytes wrong: {estimate}")
    print("PASS  QLoRA memory")


if __name__ == "__main__":
    test_single_layer_counts()
    test_total_lora_params()
    test_qlora_memory()
    print("All tests passed.")
