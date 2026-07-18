"""Reference solutions for RLAIF Reward Model."""

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))

def preference_probability(reward_winner, reward_loser):
    return sigmoid(np.asarray(reward_winner) - np.asarray(reward_loser))

def reward_model_pairwise_loss(reward_winner, reward_loser):
    p = preference_probability(reward_winner, reward_loser)
    return -np.log(p)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    p = preference_probability(3.0, 1.0)
    check(p > 0.5, "winner probability")
    print("PASS  winner probability")
    loss_good = reward_model_pairwise_loss(3.0, 1.0)
    loss_bad = reward_model_pairwise_loss(1.0, 3.0)
    check(loss_good < loss_bad, "loss ordering")
    print("PASS  loss ordering")
    print("All tests passed.")
