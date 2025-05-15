import torch
import time
from tqdm import tqdm
import multiprocessing as mp

def run_single_chain(chain_id, Y, kwargs):
    model = BSHE_voxel(Y, **kwargs)
    model.fit(chain_id, verbose=True)
    return model.get_samples()

def run_multiple_chains(Y, n_chains=4, seeds = 42, parallel = False, **kwargs):
    torch.manual_seed(seeds)

    if parallel:
        ctx = mp.get_context("spawn")  
        with ctx.Pool(processes=n_chains) as pool:
            results = pool.starmap(run_single_chain, [(i, Y, kwargs) for i in range(n_chains)])
    else:
        results=[]
        for i in range(n_chains):
            results.append(run_single_chain(i, Y, kwargs))

    # stack mcmcm samples in each chain
    results_chain = {}
    keys = results[0].keys()
    for key in keys:
        temp = [r[key] for r in results]  
        results_chain[key] = torch.stack(temp, dim=0) # dim: nchain x nsample x ...
    return results_chain

class BSHE_voxel():
    '''
    Bayesian Spatial Hierarchical Effect modeling voxel and individual level effects with independent prior
    '''

    def __init__(self, Y, 
                dtype=torch.float32, shared_var=True,
                burnin=100, thin=1, mcmc_sample=100,
                init_alpha=None,
                init_eta=None,
                init_beta=None,
                init_sig2_eta=None,
                init_sig2_beta=None,
                init_sig2_eps=None,
                ):

        self.y = Y
        self.N, self.V = Y.shape
        self.dtype = dtype
        self.shared_var = shared_var

        # mcmc settings
        self.mcmc_burnin = burnin
        self.mcmc_thinning = thin
        self.mcmc_sample = mcmc_sample
        self.total_iter = self.mcmc_burnin + self.mcmc_sample * self.mcmc_thinning

        # parameter initialization
        self.alpha = torch.randn(1, dtype=self.dtype) if init_alpha is None else init_alpha.clone()
        self.eta = torch.randn(self.N, dtype=self.dtype) if init_eta is None else init_eta.clone()
        self.eta -= self.eta.mean()
        self.beta = torch.randn(self.V, dtype=self.dtype) if init_beta is None else init_beta.clone()
        self.beta -= self.beta.mean()

        self.sig2_eta = torch.rand(1, dtype=self.dtype) if init_sig2_eta is None else init_sig2_eta.clone()
        self.sig2_beta = torch.rand(1, dtype=self.dtype) if init_sig2_beta is None else init_sig2_beta.clone()
        self.sig2_eps = torch.ones(self.V, dtype=self.dtype) if init_sig2_eps is None else init_sig2_eps.clone()

        self.sig2_alpha = 100
        self.A = 100

        self.a_eta, self.a_beta = torch.ones(2, dtype=self.dtype)
        # if each voxel share same noise prior variance or not
        if self.shared_var:
            self.a_eps = torch.ones(1, dtype=self.dtype)
            self.update_sig2_eps = self.update_sig2_eps_shared
        else:
            self.a_eps = torch.ones(self.V, dtype=self.dtype)
            self.update_sig2_eps = self.update_sig2_eps_ind

        self.update_res()
        self.loglik_y = torch.zeros(self.total_iter)
        self.make_mcmc_samples()

    def update_res(self):
        self.res = self.y - self.alpha - self.eta[:, None] - self.beta[None, :]

    def update_alpha(self):
        self.res += self.alpha
        sig2 = 1 / (self.N * (1 / self.sig2_eps).sum() + 1 / self.sig2_alpha)
        mu = sig2 * (self.res / self.sig2_eps).sum()
        self.alpha = torch.randn(1) * sig2.sqrt() + mu
        self.res -= self.alpha

    def update_eta(self):
        self.res += self.eta[:, None]
        sig2 = 1 / ((1 / self.sig2_eps).sum() + 1 / self.sig2_eta)
        mu = sig2 * (self.res / self.sig2_eps).sum(1)
        self.eta = torch.randn(self.N) * sig2.sqrt() + mu
        self.eta -= self.eta.mean()
        self.res -= self.eta[:, None]

    def update_beta(self):
        self.res += self.beta[None, :]
        sig2 = 1 / (self.N / self.sig2_eps + 1 / self.sig2_beta)
        mu = sig2 * (self.res / self.sig2_eps).sum(0)
        self.beta = torch.randn(self.V) * sig2.sqrt() + mu
        self.beta -= self.beta.mean()
        self.res -= self.beta[None, :]

    def update_sig2_eta(self):
        a_eps_new = (1 + self.N) / 2
        b_eps_new = (self.eta ** 2).sum() / 2 + 1 / self.a_eta
        self.sig2_eta = 1 / torch.distributions.Gamma(a_eps_new, b_eps_new).sample()

    def update_sig2_beta(self):
        a_eps_new = (1 + self.V) / 2
        b_eps_new = (self.beta ** 2).sum() / 2 + 1 / self.a_beta
        self.sig2_beta = 1 / torch.distributions.Gamma(a_eps_new, b_eps_new).sample()

    def update_sig2_eps_shared(self):
        a_eps_new = (1 + self.N * self.V) / 2
        b_eps_new = (self.res ** 2).sum() / 2 + 1 / self.a_eps
        sig2_eps_scalar = 1 / torch.distributions.Gamma(a_eps_new, b_eps_new).sample()
        self.sig2_eps = torch.ones_like(self.beta) * sig2_eps_scalar

        self.a_eps = 1 / torch.distributions.Gamma(1, 1 / self.A + 1 / sig2_eps_scalar).sample()

    def update_sig2_eps_ind(self):
        a_eps_new = (1 + self.N) / 2
        b_eps_new = (self.res ** 2).sum(0) / 2 + 1 / self.a_eps
        self.sig2_eps = 1 / torch.distributions.Gamma(a_eps_new, b_eps_new).sample()
        self.a_eps = 1 / torch.distributions.Gamma(1, 1 / self.A + 1 / self.sig2_eps).sample()

    def update_a(self):
        self.a_eta = 1 / torch.distributions.Gamma(1, 1 / self.A + 1 / self.sig2_eta).sample()
        self.a_beta = 1 / torch.distributions.Gamma(1, 1 / self.A + 1 / self.sig2_beta).sample()

    def update_loglik_y(self):
        return -self.N / 2 * torch.log(2 * torch.pi * self.sig2_eps).sum() - 0.5 * (self.res ** 2 / self.sig2_eps).sum()

    def make_mcmc_samples(self):
        self.mcmc_alpha = torch.zeros(self.mcmc_sample, dtype=self.dtype)
        self.mcmc_eta = torch.zeros(self.mcmc_sample, self.N, dtype=self.dtype)
        self.mcmc_beta = torch.zeros(self.mcmc_sample, self.V, dtype=self.dtype)
        self.mcmc_sig2_eta = torch.zeros(self.mcmc_sample, dtype=self.dtype)
        self.mcmc_sig2_beta = torch.zeros(self.mcmc_sample, dtype=self.dtype)
        self.mcmc_sig2_eps = torch.zeros(self.mcmc_sample, self.V, dtype=self.dtype)

    def save_mcmc_samples(self, mcmc_iter):
        self.mcmc_alpha[mcmc_iter] = self.alpha
        self.mcmc_eta[mcmc_iter, :] = self.eta
        self.mcmc_beta[mcmc_iter, :] = self.beta
        self.mcmc_sig2_eta[mcmc_iter] = self.sig2_eta
        self.mcmc_sig2_beta[mcmc_iter] = self.sig2_beta
        self.mcmc_sig2_eps[mcmc_iter, :] = self.sig2_eps

    def fit(self, chain_id=0, verbose=False):
        start_time = time.time()
        for i in tqdm(range(self.total_iter)):
            self.update_alpha()
            self.update_eta()
            self.update_beta()
            self.update_sig2_eta()
            self.update_sig2_beta()
            self.update_sig2_eps()
            self.update_a()
            self.loglik_y[i] = self.update_loglik_y()
            if i >= self.mcmc_burnin and (i - self.mcmc_burnin) % self.mcmc_thinning == 0:
                mcmc_iter = int((i - self.mcmc_burnin) / self.mcmc_thinning)
                self.save_mcmc_samples(mcmc_iter)
        self.runtime = time.time() - start_time
        if verbose:
            print(f"Chain {chain_id + 1} finished in {self.runtime:.2f} seconds")

    def get_samples(self):
        return {
            "alpha": self.mcmc_alpha,
            "eta": self.mcmc_eta,
            "beta": self.mcmc_beta,
            "sig2_eta": self.mcmc_sig2_eta,
            "sig2_beta": self.mcmc_sig2_beta,
            "sig2_eps": self.mcmc_sig2_eps,
            "loglik": self.loglik_y
        }
    
def PPC(data, samples, n_samples=100, dtype=torch.float32):
    """
    Posterior Predictive Check (PPC)

    Args:
        n_sample (int): number of posterior samples
        dtype (torch.dtype): output tensor dtype

    Returns:
        pred_y: tensor of shape (n_sample, N, V)
    """
    n_chains, n_mcmc = samples['alpha'].shape
    N, V = data.shape
    if n_samples > n_mcmc:
        print("number of draws larger than mcmc samples")
    idx = torch.linspace(0, n_mcmc - 1, n_samples, dtype=torch.int32)
    pred_y = torch.zeros(n_chains, n_samples, N, V, dtype=dtype)
    for s in range(n_chains):
        for i, ind in enumerate(idx):
            alpha = samples['alpha'][s, ind]
            eta = samples['eta'][s, ind]
            beta = samples['beta'][s, ind]
            sig2_eps = samples['sig2_eps'][s, ind]

            mean = alpha + eta[:, None] + beta[None, :]
            noise = torch.randn(N, V) * sig2_eps.sqrt()
            pred_y[s, i] = mean + noise
    return pred_y


def get_ll(data, samples, dtype=torch.float32):
    """
    get log-likelihood for each observation (voxel-wise)

    Args:
        data (tensor): input data
        samples (dic): dictionary contains mcmc samples for model 1

    Returns:
        pred_y: tensor of shape (n_sample, N, V)
    """
    n_chains, n_mcmc = samples['alpha'].shape
    N, V = data.shape

    ll_mat = torch.zeros(n_chains, n_mcmc, V,  dtype=dtype)
    for s in range(n_chains):
        for ind in range(n_mcmc):
            alpha = samples['alpha'][s, ind].to(dtype=dtype)
            eta = samples['eta'][s, ind].to(dtype=dtype)
            beta = samples['beta'][s, ind].to(dtype=dtype)
            sig2_eps = samples['sig2_eps'][s, ind].to(dtype=dtype)

            mu = alpha + eta[:, None] + beta[None, :]
            res = data - mu
            ll = -0.5 * torch.log(2 * torch.pi * sig2_eps) - 0.5 * (res**2) / sig2_eps
            #ll_mat[s,ind] = ll.sum(dim=1) # individual level
            ll_mat[s,ind] = ll.sum(dim=0)
    return ll_mat
    


