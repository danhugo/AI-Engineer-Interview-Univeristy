"""Practice for LightGBM Objective Function."""

def leaf_score(grad_sum,hess_sum,l2):
    raise NotImplementedError

def split_gain(g_left,h_left,g_right,h_right,l2=0.0,min_gain_to_split=0.0):
    raise NotImplementedError

def histogram_gradient_sums(feature_bins,gradients,hessians,n_bins):
    raise NotImplementedError
