"""Reference solutions for XGBoost Objective Function."""

import numpy as np

def xgboost_leaf_weight(gradients,hessians,reg_lambda):
    G=float(np.sum(gradients)); H=float(np.sum(hessians)); return -G/(H+reg_lambda)

def xgboost_leaf_score(gradients,hessians,reg_lambda):
    G=float(np.sum(gradients)); H=float(np.sum(hessians)); return .5*G*G/(H+reg_lambda)

def xgboost_split_gain(g_left,h_left,g_right,h_right,reg_lambda,gamma):
    left=xgboost_leaf_score(g_left,h_left,reg_lambda); right=xgboost_leaf_score(g_right,h_right,reg_lambda)
    parent=xgboost_leaf_score(np.r_[g_left,g_right],np.r_[h_left,h_right],reg_lambda); return float(left+right-parent-gamma)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    w=xgboost_leaf_weight([-2,-1],[1,1],1); check(np.isclose(w,1),"leaf weight"); print("PASS  leaf weight")
    s=xgboost_leaf_score([-2,-1],[1,1],1); check(np.isclose(s,1.5),"leaf score"); print("PASS  leaf score")
    g=xgboost_split_gain([-2],[1],[2],[1],1,0); check(np.isclose(g,2.0),"split gain"); print("PASS  split gain")
    print("All tests passed.")
