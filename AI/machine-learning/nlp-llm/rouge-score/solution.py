"""Reference solutions for ROUGE Score."""

from collections import Counter
import re


def tokenize(text):
    """Lowercase and split text into word tokens."""
    return re.findall(r"\w+", text.lower())


def ngrams(tokens, n):
    """Return n-grams as tuples."""
    if n <= 0:
        raise ValueError("n must be positive")
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n(candidate, reference, n=1):
    """Return (precision, recall, f1) for ROUGE-N."""
    cand_ngrams = Counter(ngrams(tokenize(candidate), n))
    ref_ngrams = Counter(ngrams(tokenize(reference), n))
    cand_total = sum(cand_ngrams.values())
    ref_total = sum(ref_ngrams.values())
    if cand_total == 0 or ref_total == 0:
        return 0.0, 0.0, 0.0

    overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
    precision = overlap / cand_total
    recall = overlap / ref_total
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def lcs_length(a, b):
    """Return the longest common subsequence length for two token lists."""
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(candidate, reference):
    """Return (precision, recall, f1) for ROUGE-L."""
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    lcs = lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


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
