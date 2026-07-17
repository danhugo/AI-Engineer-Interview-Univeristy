"""AI / Machine Learning - N-Gram Speculation Dictionary Practice."""


def build_ngram_dictionary(tokens, min_n, max_n, max_continuation):
    """Map each n-gram tuple to the first continuation that followed it."""
    pass


def propose_from_dictionary(tokens_so_far, table, min_n, max_n, max_tokens):
    """Return the longest-key matching continuation, capped to max_tokens."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_dictionary_construction():
    tokens = ["a", "b", "c", "a", "b", "d", "a", "b", "e"]
    table = build_ngram_dictionary(tokens, min_n=2, max_n=3, max_continuation=2)

    check(table[("a", "b")] == ["c", "a"], "first continuation should be stored")
    check(table[("b", "c", "a")] == ["b", "d"], "3-gram continuation should be stored")
    print("PASS  dictionary construction")


def test_longest_match_proposal():
    tokens = ["the", "cat", "sat", "the", "cat", "slept"]
    table = build_ngram_dictionary(tokens, min_n=1, max_n=2, max_continuation=2)

    proposal = propose_from_dictionary(["x", "the", "cat"], table, 1, 2, 2)
    check(proposal == ["sat", "the"], "should prefer the 2-gram match")

    proposal = propose_from_dictionary(["unknown"], table, 1, 2, 2)
    check(proposal == [], "unknown suffix should produce no proposal")
    print("PASS  proposal lookup")


if __name__ == "__main__":
    test_dictionary_construction()
    test_longest_match_proposal()
    print("All tests passed.")
