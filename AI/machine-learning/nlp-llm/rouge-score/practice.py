"""AI / Machine Learning - ROUGE Score Practice."""

from collections import Counter
import re


def tokenize(text):
    """Lowercase and split text into word tokens."""
    pass


def ngrams(tokens, n):
    """Return n-grams as tuples."""
    pass


def rouge_n(candidate, reference, n=1):
    """Return (precision, recall, f1) for ROUGE-N."""
    pass


def lcs_length(a, b):
    """Return the longest common subsequence length for two token lists."""
    pass


def rouge_l(candidate, reference):
    """Return (precision, recall, f1) for ROUGE-L."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close_tuple(actual, expected, tol=1e-6):
    return all(abs(float(a) - float(e)) < tol for a, e in zip(actual, expected))


def test_rouge_n():
    cand = "the cat sat on the mat"
    ref = "the cat is on the mat"
    r1 = rouge_n(cand, ref, n=1)
    r2 = rouge_n(cand, ref, n=2)
    check(close_tuple(r1, (5 / 6, 5 / 6, 5 / 6)), f"ROUGE-1 wrong: {r1}")
    check(close_tuple(r2, (3 / 5, 3 / 5, 3 / 5)), f"ROUGE-2 wrong: {r2}")
    print("PASS  ROUGE-N")


def test_lcs_and_rouge_l():
    a = tokenize("the cat sat on the mat")
    b = tokenize("the cat is on mat")
    check(lcs_length(a, b) == 4, "LCS length should allow gaps")
    score = rouge_l("the cat sat on the mat", "the cat is on mat")
    check(close_tuple(score, (4 / 6, 4 / 5, 8 / 11)), f"ROUGE-L wrong: {score}")
    print("PASS  ROUGE-L")


def test_repetition_is_clipped():
    score = rouge_n("cat cat cat cat", "cat sat", n=1)
    check(close_tuple(score, (1 / 4, 1 / 2, 1 / 3)), f"clipped overlap wrong: {score}")
    print("PASS  clipped counts")


if __name__ == "__main__":
    test_rouge_n()
    test_lcs_and_rouge_l()
    test_repetition_is_clipped()
    print("All tests passed.")
