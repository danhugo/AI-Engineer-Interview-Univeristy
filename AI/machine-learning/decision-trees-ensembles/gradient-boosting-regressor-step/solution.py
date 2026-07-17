"""Reference solutions for Gradient Boosting Regressor Step."""

import numpy as np

def squared_error_residuals(y_true,y_pred):
    return np.asarray(y_true,float)-np.asarray(y_pred,float)

def update_predictions(y_pred,tree_pred,learning_rate):
    return np.asarray(y_pred,float)+learning_rate*np.asarray(tree_pred,float)

def fit_constant_to_residuals(residuals):
    return float(np.asarray(residuals,float).mean())

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    y=np.array([2.,4.,6.]); p=np.array([3.,3.,3.]); r=squared_error_residuals(y,p); check(np.allclose(r,[-1,1,3]),"residuals"); print("PASS  residuals")
    c=fit_constant_to_residuals(r); check(np.isclose(c,1),"constant residual fit"); print("PASS  constant residual fit")
    p1=update_predictions(p,np.full_like(p,c),.1); check(np.allclose(p1,[3.1,3.1,3.1]),"prediction update"); print("PASS  prediction update")
    print("All tests passed.")
