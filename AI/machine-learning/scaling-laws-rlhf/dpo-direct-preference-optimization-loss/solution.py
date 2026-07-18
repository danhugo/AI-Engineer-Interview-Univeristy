"""Reference solutions for DPO Direct Preference Optimization Loss."""

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))

def dpo_margin(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta):
    return beta * ((policy_logp_w - ref_logp_w) - (policy_logp_l - ref_logp_l))

def dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta):
    margin = dpo_margin(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta)
    return -np.log(sigmoid(margin))

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    m = dpo_margin(-1.0, -3.0, -2.0, -2.0, beta=0.5)
    check(np.isclose(m, 1.0), "margin")
    print("PASS  margin")
    good = dpo_loss(-1.0, -3.0, -2.0, -2.0, beta=0.5)
    bad = dpo_loss(-3.0, -1.0, -2.0, -2.0, beta=0.5)
    check(good < bad, "loss ordering")
    print("PASS  loss ordering")
    print("All tests passed.")
