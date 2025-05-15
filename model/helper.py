import numpy as np
import torch
from model.utils import save_pickle 



def matern_kernel(X, Y, lengthscale=1.0, variance=1.0, nu=1.5):
    pairwise_dists = torch.cdist(X, Y, p=2)  # Euclidean distance
    sqrt3 = torch.sqrt(torch.tensor(3.0))
    sqrt5 = torch.sqrt(torch.tensor(5.0))
    
    if nu == 0.5:
        K = torch.exp(-pairwise_dists / lengthscale)  # Exponential kernel
    elif nu == 1.5:
        K = (1.0 + sqrt3 * pairwise_dists / lengthscale) * torch.exp(-sqrt3 * pairwise_dists / lengthscale)
    elif nu == 2.5:
        K = (1.0 + sqrt5 * pairwise_dists / lengthscale + (5.0/3.0) * (pairwise_dists / lengthscale)**2) * torch.exp(-sqrt5 * pairwise_dists / lengthscale)
    else:
        raise ValueError("Currently supports only nu = 0.5, 1.5, 2.5")
    
    return variance * K

def ker_approx(ker_func, S, c, dtype=torch.float32):
    '''
    kernel: kernel function
    x: original coordiantes
    c: inducing coordiantes
    '''
    dtype = dtype
    Kc = torch.from_numpy(ker_func(c, c)).to(dtype=dtype)
    Ksc = torch.from_numpy(ker_func(S, c)).to(dtype=dtype)

    R = torch.linalg.cholesky(Kc)
    R_inv = torch.linalg.inv(R.T)
    KR = (Ksc @ R_inv)
    U, D, _ = torch.linalg.svd(KR, full_matrices=False)
    return U, D
