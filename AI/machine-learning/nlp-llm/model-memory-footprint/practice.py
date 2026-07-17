"""
AI / Machine Learning — Model Memory Footprint
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""


DTYPE_BYTES = {
    "float32": 4.0,
    "bfloat16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def parameter_memory_bytes(num_parameters, dtype):
    """Return raw parameter memory in bytes."""
    pass


def bytes_to_gib(num_bytes):
    """Convert bytes to binary GiB."""
    pass


def adam_training_state_bytes(num_parameters, param_dtype="float16", grad_dtype="float16", optimizer_dtype="float32"):
    """
    Estimate parameter + gradient + Adam first/second moment memory.

    HINT:
      Adam has two optimizer state tensors per parameter.
    """
    pass


def kv_cache_bytes(num_layers, batch_size, sequence_length, hidden_size, dtype="float16"):
    """
    Estimate decoder-only KV cache bytes.

    HINT:
      key and value are both cached.
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_parameter_memory_bytes():
    check(parameter_memory_bytes(7_000_000_000, "float16") == 14_000_000_000, "float16 memory wrong")
    check(parameter_memory_bytes(1_000, "int4") == 500, "int4 memory wrong")
    print("PASS  parameter_memory_bytes")


def test_bytes_to_gib():
    check(round(bytes_to_gib(2**30), 4) == 1.0, "GiB conversion wrong")
    check(round(bytes_to_gib(14_000_000_000), 3) == 13.039, "14B bytes GiB wrong")
    print("PASS  bytes_to_gib")


def test_adam_training_state_bytes():
    total = adam_training_state_bytes(1_000_000, "float16", "float16", "float32")
    check(total == 12_000_000, f"Adam estimate wrong: {total}")
    print("PASS  adam_training_state_bytes")


def test_kv_cache_bytes():
    total = kv_cache_bytes(num_layers=32, batch_size=2, sequence_length=1024, hidden_size=4096, dtype="float16")
    check(total == 1_073_741_824, f"KV cache estimate wrong: {total}")
    print("PASS  kv_cache_bytes")


if __name__ == "__main__":
    test_parameter_memory_bytes()
    test_bytes_to_gib()
    test_adam_training_state_bytes()
    test_kv_cache_bytes()
    print("All tests passed.")
