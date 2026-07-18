"""Reference solutions for Elbow Method for K Selection."""

import numpy as np

def inertia_improvements(inertias):
    inertias=np.asarray(inertias,float)
    return inertias[:-1] - inertias[1:]

def elbow_by_relative_drop(inertias, threshold=0.5):
    improvements=inertia_improvements(inertias)
    if len(improvements) < 2: return 1
    ratios=improvements[1:] / np.maximum(improvements[:-1], 1e-12)
    for i, r in enumerate(ratios, start=2):
        if r < threshold:
            return i
    return len(inertias)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    inertias=[100,50,30,25,23]
    imp=inertia_improvements(inertias); check(np.array_equal(imp,[50,20,5,2]),"improvements"); print("PASS  improvements")
    check(elbow_by_relative_drop(inertias, threshold=.4)==3,"elbow k"); print("PASS  elbow k")
    print("All tests passed.")
