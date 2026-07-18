"""Reference solutions for PPO Clipped Objective."""

import numpy as np

def probability_ratio(new_logprob, old_logprob):
    return np.exp(np.asarray(new_logprob, dtype=float) - np.asarray(old_logprob, dtype=float))

def ppo_clipped_terms(ratio, advantage, epsilon):
    ratio = np.asarray(ratio, dtype=float)
    advantage = np.asarray(advantage, dtype=float)
    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
    return unclipped, clipped, np.minimum(unclipped, clipped)

def ppo_clipped_objective(new_logprob, old_logprob, advantage, epsilon):
    ratio = probability_ratio(new_logprob, old_logprob)
    return float(np.mean(ppo_clipped_terms(ratio, advantage, epsilon)[2]))

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    r = probability_ratio(np.log([1.2, 0.7]), np.log([1.0, 1.0]))
    check(np.allclose(r, [1.2,0.7]), "ratio")
    print("PASS  ratio")
    _, _, term = ppo_clipped_terms([1.5,0.5], [1.0,-1.0], epsilon=0.2)
    check(np.allclose(term, [1.2,-0.8]), "clipped terms")
    print("PASS  clipped terms")
    print("All tests passed.")
