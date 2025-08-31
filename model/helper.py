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

def ker_approx(kernel, grids, grids_c, err = 1e-4, dtype=torch.float32):
    '''
    Get low-rank approxiamted kernel using inducing points

    Args:
        kernel (sklearn.gaussian_process.kernels): kernel objects
        grids (torch.tensor): spatial locations
        grids_c: inducing coordiantes
    Returns:
        U: tensor of basis functions (V, L)
        D: tensor of square root of eigenvalues associated to the basis (L)
    '''
    dtype = dtype
    Kc = torch.from_numpy(kernel(grids_c, grids_c)).to(dtype=dtype)
    Kc  += torch.eye(Kc.shape[0]) * err
    Ksc = torch.from_numpy(kernel(grids, grids_c)).to(dtype=dtype)

    R = torch.linalg.cholesky(Kc)
    R_inv = torch.linalg.inv(R.T)
    KR = (Ksc @ R_inv)
    U, D, _ = torch.linalg.svd(KR, full_matrices=False)
    return U, D

def get_basis(grids, L, kernel, err = 1e-4, dtype=torch.float32):
    """
    Get basis functions from low-rank approximated kernel

    Args:
        grids (torch.tensor): d-dimensional spatial coordinates (V, L)
        L (int): number of basis used for kernel approximation (tuning parameter)
        kernel (sklearn.gaussian_process.kernels): kernel objects
        err (float): a small jitter to ensure valid cholesky decomposition
        dtype (torch.dtype): output tensor dtype

    Returns:
        B: tensor of basis functions (V, L)
        eig_val_sqrt: tensor of square root of eigenvalues associated to the basis (L)
    """
    V = grids.shape[0]
    # select equally spaced inducing points
    induce = torch.linspace(0, V, steps=L+2)[1:-1].long()
    grids_c = grids[induce,:]
    B, eig_val_sqrt = ker_approx(kernel, grids, grids_c, err = err, dtype = dtype)
    return B, eig_val_sqrt