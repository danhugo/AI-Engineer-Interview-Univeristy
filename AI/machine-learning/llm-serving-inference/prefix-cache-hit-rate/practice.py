"""
AI / Machine Learning - Prefix Cache Hit Rate
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""


def common_prefix_length(a, b):
    """Return the number of matching tokens from the start of two sequences."""
    pass


def reusable_full_blocks(common_prefix_tokens, block_size):
    """Return how many complete KV blocks can be reused."""
    pass


def block_hit_rate(reused_blocks, requested_prompt_tokens, block_size):
    """Return reused full blocks divided by requested full prompt blocks."""
    pass


def token_equivalent_hit_rate(reused_blocks, requested_prompt_tokens, block_size):
    """Return the fraction of prompt tokens whose prefill work is saved."""
    pass


def best_prefix_cache_match(request_tokens, cached_prompts, block_size):
    """
    Return (best_index, reused_blocks) for the cached prompt with the longest
    reusable full-block prefix.
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_common_prefix_length():
    check(common_prefix_length([1, 2, 3], [1, 2, 9]) == 2, "common prefix wrong")
    check(common_prefix_length([1], [2]) == 0, "zero prefix wrong")
    print("PASS  common_prefix_length")


def test_reusable_full_blocks():
    check(reusable_full_blocks(31, 16) == 1, "31 tokens should reuse one block")
    check(reusable_full_blocks(32, 16) == 2, "32 tokens should reuse two blocks")
    print("PASS  reusable_full_blocks")


def test_hit_rates():
    check(block_hit_rate(2, 40, 16) == 2 / 2, "block hit rate wrong")
    check(token_equivalent_hit_rate(2, 40, 16) == 32 / 40, "token hit rate wrong")
    check(block_hit_rate(0, 7, 16) == 0.0, "short prompt block rate wrong")
    print("PASS  hit_rates")


def test_best_prefix_cache_match():
    cached = [
        [10, 11, 12, 13, 14, 15, 16, 17],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 99, 99, 99, 99],
    ]
    request = [1, 2, 3, 4, 5, 6, 7, 8, 42]
    check(best_prefix_cache_match(request, cached, block_size=4) == (1, 2), "best cache match wrong")
    print("PASS  best_prefix_cache_match")


if __name__ == "__main__":
    test_common_prefix_length()
    test_reusable_full_blocks()
    test_hit_rates()
    test_best_prefix_cache_match()
    print("All tests passed.")
