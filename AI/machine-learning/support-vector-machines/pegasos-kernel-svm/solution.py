"""Reference solutions for Pegasos Kernel SVM."""

import numpy as np

def pegasos_step_size(reg_lambda, t):
    return 1.0 / (reg_lambda * t)

def pegasos_linear_step(w, x_i, y_i, reg_lambda, t):
    w = np.asarray(w, dtype=float)
    x_i = np.asarray(x_i, dtype=float)
    eta = pegasos_step_size(reg_lambda, t)
    if y_i * np.dot(w, x_i) < 1.0:
        return (1 - eta * reg_lambda) * w + eta * y_i * x_i
    return (1 - eta * reg_lambda) * w

def project_to_pegasos_ball(w, reg_lambda):
    w = np.asarray(w, dtype=float)
    radius = 1.0 / np.sqrt(reg_lambda)
    norm = np.linalg.norm(w)
    return w if norm <= radius else w * (radius / norm)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    check(np.isclose(pegasos_step_size(.5, 4), .5), "step size")
    print("PASS  step size")
    w = pegasos_linear_step([0,0], [2,0], 1, reg_lambda=.5, t=1)
    check(np.allclose(w, [4,0]), "violating update")
    print("PASS  violating update")
    p = project_to_pegasos_ball([10,0], reg_lambda=.25)
    check(np.isclose(np.linalg.norm(p), 2.0), "projection")
    print("PASS  projection")
    print("All tests passed.")
