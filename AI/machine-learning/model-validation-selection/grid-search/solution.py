"""Reference solutions for Grid Search."""

import itertools

def parameter_grid(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

def select_best_result(results, higher_is_better=True):
    key = (lambda r: r["score"]) if higher_is_better else (lambda r: -r["score"])
    return max(results, key=key)

def check(c, m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    combos = parameter_grid({"depth": [2, 3], "lr": [0.1, 0.01]})
    check(len(combos) == 4, "combination count")
    print("PASS  combination count")
    check({"depth": 2, "lr": 0.1} in combos, "contains expected combo")
    print("PASS  contains expected combo")
    best = select_best_result([{"params": {"a": 1}, "score": 0.7}, {"params": {"a": 2}, "score": 0.9}])
    check(best["params"]["a"] == 2, "best result")
    print("PASS  best result")
    print("All tests passed.")
