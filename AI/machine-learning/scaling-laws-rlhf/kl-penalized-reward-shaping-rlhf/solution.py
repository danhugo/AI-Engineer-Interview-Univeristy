"""Reference solutions for KL-Penalized Reward Shaping RLHF."""

import numpy as np

def categorical_kl(policy_probs, reference_probs):
    p = np.asarray(policy_probs, dtype=float)
    q = np.asarray(reference_probs, dtype=float)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def sampled_kl_from_logprobs(policy_logprobs, reference_logprobs):
    return np.asarray(policy_logprobs, dtype=float) - np.asarray(reference_logprobs, dtype=float)

def shaped_reward(reward_model_score, sampled_kl, beta):
    return np.asarray(reward_model_score, dtype=float) - beta * np.asarray(sampled_kl, dtype=float)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    kl = categorical_kl([0.5,0.5], [0.25,0.75])
    check(kl > 0, "positive kl")
    print("PASS  positive kl")
    sk = sampled_kl_from_logprobs([-1.0,-2.0], [-1.5,-1.5])
    check(np.allclose(sk, [0.5,-0.5]), "sampled kl")
    print("PASS  sampled kl")
    sr = shaped_reward([2.0,2.0], sk, beta=0.2)
    check(np.allclose(sr, [1.9,2.1]), "shaped reward")
    print("PASS  shaped reward")
    print("All tests passed.")
