"""AI / Machine Learning - BLEU Score Practice."""

from collections import Counter
import math
import re


def tokenize(text):
    """Lowercase and split text into word tokens."""
    pass


def ngrams(tokens, n):
    """Return n-grams as tuples."""
    pass


def modified_precision(candidate_tokens, reference_tokens_list, n):
    """Return BLEU clipped n-gram precision."""
    pass


def closest_reference_length(candidate_len, reference_lens):
    """Return the reference length closest to candidate_len; tie goes shorter."""
    pass


def brevity_penalty(candidate_len, reference_len):
    """Return BLEU brevity penalty."""
    pass


def sentence_bleu(candidate, references, max_n=4, smooth=False):
    """Return sentence BLEU for one candidate and one or more references."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


def test_modified_precision():
    cand = tokenize("the the the the")
    refs = [tokenize("the cat is on the mat")]
    check(close(modified_precision(cand, refs, 1), 2 / 4), "unigram clipping failed")
    check(close(modified_precision(tokenize("the cat sat"), refs, 2), 1 / 2), "bigram precision failed")
    print("PASS  modified precision")


def test_brevity_penalty():
    check(close(brevity_penalty(5, 5), 1.0), "equal length BP should be 1")
    check(close(brevity_penalty(4, 6), math.exp(1 - 6 / 4)), "short candidate BP wrong")
    check(closest_reference_length(5, [3, 7, 6]) == 6, "closest reference length wrong")
    print("PASS  brevity penalty")


def test_sentence_bleu():
    score = sentence_bleu(
        "the cat is on the mat",
        ["the cat is on the mat"],
        max_n=4,
    )
    check(close(score, 1.0), f"perfect BLEU should be 1: {score}")

    short = sentence_bleu("the cat", ["the cat is on the mat"], max_n=2)
    check(close(short, math.exp(1 - 6 / 2)), f"brevity penalty not applied: {short}")

    smoothed = sentence_bleu("the cat sat", ["the dog barked"], max_n=4, smooth=True)
    check(smoothed > 0.0, f"smoothed BLEU should be positive: {smoothed}")
    print("PASS  sentence BLEU")


if __name__ == "__main__":
    test_modified_precision()
    test_brevity_penalty()
    test_sentence_bleu()
    print("All tests passed.")
