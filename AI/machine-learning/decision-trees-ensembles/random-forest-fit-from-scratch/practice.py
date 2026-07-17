"""Practice for Random Forest Fit from Scratch."""

def bootstrap_indices(n_samples, seed=None):
    raise NotImplementedError

def gini(y):
    raise NotImplementedError

def majority(y):
    raise NotImplementedError

def fit_random_stump(X,y,features):
    raise NotImplementedError

def predict_stump(stump,X):
    raise NotImplementedError

def fit_random_stump_forest(X,y,n_estimators=10,max_features=1,seed=0):
    raise NotImplementedError

def predict_random_stump_forest(forest,X):
    raise NotImplementedError
