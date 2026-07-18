"""Reference solutions for Compute-Optimal Model Size IsoFLOP."""

import numpy as np

def training_compute_flops(parameters, tokens):
    return 6.0 * parameters * tokens

def tokens_for_isoflop(compute, parameters):
    return compute / (6.0 * parameters)

def chinchilla_rule_tokens(parameters, tokens_per_parameter=20.0):
    return parameters * tokens_per_parameter

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    C = training_compute_flops(1e9, 2e10)
    check(np.isclose(C, 1.2e20), "compute")
    print("PASS  compute")
    D = tokens_for_isoflop(1.2e20, 1e9)
    check(np.isclose(D, 2e10), "isoflop tokens")
    print("PASS  isoflop tokens")
    check(np.isclose(chinchilla_rule_tokens(7e9), 1.4e11), "rule tokens")
    print("PASS  rule tokens")
    print("All tests passed.")
