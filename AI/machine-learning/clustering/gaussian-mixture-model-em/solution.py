"""Reference solutions for Gaussian Mixture Model with EM."""

import numpy as np

def gaussian_pdf_1d(x, mean, var):
    x=np.asarray(x,float)
    return np.exp(-0.5*((x-mean)**2)/var) / np.sqrt(2*np.pi*var)

def gmm_responsibilities_1d(x, weights, means, variances):
    comps=np.vstack([weights[k]*gaussian_pdf_1d(x,means[k],variances[k]) for k in range(len(weights))]).T
    return comps / comps.sum(axis=1, keepdims=True)

def gmm_update_means_1d(x, responsibilities):
    x=np.asarray(x,float); r=np.asarray(responsibilities,float)
    return (r*x[:,None]).sum(axis=0) / r.sum(axis=0)

def check(c,m):
    if not c: raise AssertionError(f"FAIL  {m}")

if __name__ == "__main__":
    r=gmm_responsibilities_1d([0,10],[.5,.5],[0,10],[1,1]); check(np.allclose(r.sum(axis=1),[1,1]),"responsibility sums"); print("PASS  responsibility sums")
    check(r[0,0]>r[0,1] and r[1,1]>r[1,0],"component preference"); print("PASS  component preference")
    means=gmm_update_means_1d([0,10], [[1,0],[0,1]]); check(np.allclose(means,[0,10]),"mean update"); print("PASS  mean update")
    print("All tests passed.")
