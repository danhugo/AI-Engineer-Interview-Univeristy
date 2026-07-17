"""Reference solutions for Random Forest Feature Importance."""

import numpy as np

def impurity_decrease(parent_impurity,left_impurity,right_impurity,n_parent,n_left,n_right,n_total):
    child=(n_left/n_parent)*left_impurity+(n_right/n_parent)*right_impurity
    return float((n_parent/n_total)*(parent_impurity-child))

def normalized_feature_importances(splits,n_features):
    scores=np.zeros(n_features,float)
    for s in splits:
        scores[s["feature"]]+=impurity_decrease(s["parent_impurity"],s["left_impurity"],s["right_impurity"],s["n_parent"],s["n_left"],s["n_right"],s["n_total"])
    return scores/scores.sum() if scores.sum()>0 else scores

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    dec=impurity_decrease(.5,0,0,10,5,5,10); check(np.isclose(dec,.5),"weighted decrease"); print("PASS  weighted decrease")
    splits=[{"feature":0,"parent_impurity":.5,"left_impurity":0,"right_impurity":0,"n_parent":10,"n_left":5,"n_right":5,"n_total":10},{"feature":1,"parent_impurity":.4,"left_impurity":.2,"right_impurity":.2,"n_parent":5,"n_left":2,"n_right":3,"n_total":10}]
    imp=normalized_feature_importances(splits,3); check(np.isclose(imp.sum(),1) and imp[0]>imp[1]>imp[2],"normalized importances"); print("PASS  normalized importances")
    print("All tests passed.")
