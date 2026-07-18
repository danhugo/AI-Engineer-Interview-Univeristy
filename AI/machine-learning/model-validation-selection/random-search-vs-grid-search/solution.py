"""Reference solutions for Random Search vs Grid Search."""

import numpy as np

def sample_random_search(space, n_iter, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_iter):
        params = {}
        for name, spec in space.items():
            if spec["type"] == "choice":
                params[name] = rng.choice(spec["values"]).item() if hasattr(rng.choice(spec["values"]), "item") else rng.choice(spec["values"])
            elif spec["type"] == "uniform":
                params[name] = float(rng.uniform(spec["low"], spec["high"]))
            elif spec["type"] == "loguniform":
                params[name] = float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
            else:
                raise ValueError("unknown distribution")
        samples.append(params)
    return samples

def grid_size(grid):
    total = 1
    for values in grid.values():
        total *= len(values)
    return total

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    check(grid_size({"a": [1, 2], "b": [3, 4, 5]}) == 6, "grid size")
    print("PASS  grid size")
    samples = sample_random_search({"lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1}, "depth": {"type": "choice", "values": [2, 3]}}, 5, seed=2)
    check(len(samples) == 5, "sample count")
    print("PASS  sample count")
    check(all(1e-4 <= s["lr"] <= 1e-1 for s in samples), "sample bounds")
    print("PASS  sample bounds")
    print("All tests passed.")
