"""Reference solutions for Power-Law Scaling Law."""

import numpy as np

def power_law_loss(x, irreducible_loss, coefficient, exponent):
    return irreducible_loss + coefficient * np.asarray(x, dtype=float) ** (-exponent)

def estimate_exponent_from_two_points(x1, loss1, x2, loss2, irreducible_loss):
    y1 = loss1 - irreducible_loss
    y2 = loss2 - irreducible_loss
    return float(-np.log(y2 / y1) / np.log(x2 / x1))

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    losses = power_law_loss([1, 4], 1.0, 2.0, 0.5)
    check(np.allclose(losses, [3.0, 2.0]), "loss values")
    print("PASS  loss values")
    a = estimate_exponent_from_two_points(1, 3.0, 4, 2.0, 1.0)
    check(np.isclose(a, 0.5), "exponent estimate")
    print("PASS  exponent estimate")
    print("All tests passed.")
