"""Reference solutions for Combined RM and LLM Quality Filter."""

import numpy as np

def combined_quality_score(rm_score, judge_score, risk_score, weights=(0.6, 0.4, 1.0)):
    wrm, wjudge, wrisk = weights
    return wrm * np.asarray(rm_score) + wjudge * np.asarray(judge_score) - wrisk * np.asarray(risk_score)

def passes_quality_filter(rm_score, judge_score, risk_score, min_score=0.0, max_risk=0.5):
    score = combined_quality_score(rm_score, judge_score, risk_score)
    return np.logical_and(score >= min_score, np.asarray(risk_score) <= max_risk)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    s = combined_quality_score([1.0,1.0], [1.0,0.0], [0.0,1.0])
    check(np.allclose(s, [1.0,-0.4]), "combined score")
    print("PASS  combined score")
    keep = passes_quality_filter([1.0,1.0], [1.0,0.0], [0.0,1.0], min_score=0)
    check(np.array_equal(keep, [True, False]), "quality filter")
    print("PASS  quality filter")
    print("All tests passed.")
