"""Reference solutions for Bayesian Hyperparameter Optimization."""

import numpy as np
from math import erf

def expected_improvement(mu, sigma, best, xi=0.0):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    improvement = mu - best - xi
    z = np.divide(improvement, sigma, out=np.zeros_like(improvement), where=sigma > 0)
    normal_pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    normal_cdf = 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2)))
    ei = improvement * normal_cdf + sigma * normal_pdf
    return np.where(sigma > 0, ei, np.maximum(improvement, 0.0))

def choose_next_by_acquisition(candidates, acquisition_values):
    idx = int(np.argmax(acquisition_values))
    return candidates[idx], idx

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    ei = expected_improvement([0.8, 0.6], [0.1, 0.5], best=0.7)
    check(ei[0] > 0 and ei[1] > 0, "positive EI")
    print("PASS  positive EI")
    cand, idx = choose_next_by_acquisition([{"lr": 0.1}, {"lr": 0.01}], [0.2, 0.5])
    check(idx == 1 and cand["lr"] == 0.01, "choose max acquisition")
    print("PASS  choose max acquisition")
    print("All tests passed.")
