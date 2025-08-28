import numpy             as np
#[YNS] : which version of pytorch was the code developed on 
import torch
import nibabel           as nib
import matplotlib.pyplot as plt
import importlib
#from   model.utils      import save_pickle, load_pickle
from model.utils import save_pickle, load_pickle
from   nilearn.plotting import plot_stat_map
from   sklearn.gaussian_process.kernels import RBF, Matern
from   fastkde import fastKDE
import argparse             as argp
import os

import sys
import model.BSHE_Gibbs as BSHE_Gibbs
import model.BSHE_VI    as BSHE_VI



list_model = [ 'BSHE_Gibbs','BSHE_VI']



def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def load_data(data_path):

    template_path = os.path.join(
        data_path, 'group_analysis.ttest.1grp.equalRange.gain','MNI152_2009_template_SSW.nii.gz')
    mni_img = nib.load(str(template_path))
    mni_data = mni_img.get_fdata()[:, :, :, 0, 0]  # Extract the 0-th volume
    target_affine = mni_img.affine
    x,y,z=(mni_data).nonzero()
    
    return x,y,z

def load_mask_data(mask_path):
    maskfl_path = os.path.join(
        mask_path, 'group_analysis.ttest.1grp.equalRange.gain','group_mask.inter.nii.gz')
    mask_img = nib.Nifti1Image.from_filename(maskfl_path)
    mask = nib.Nifti1Image.from_filename(maskfl_path).get_fdata()

    return mask 
'''
Since importlib.reload() requires a module object, 
not a string, you must first retrieve the module from 
sys.modules using the string name provided by the command-line argument. 
'''

def import_model_by_name(module_name: str, package: str | None = "model"):
  
    
    # Build a fully-qualified module path if a package is provided
    fqmn = module_name if package is None else f"{package}.{module_name}"

    # If already imported, reload the existing module object
    if fqmn in sys.modules:
        return importlib.reload(sys.modules[fqmn])

    # Otherwise import it fresh
    return importlib.import_module(fqmn)


def get_stat_map():
    post_map = np.zeros(mask_img.shape)
    post_map[x, y, z] = voxel_mcmc[0]
    post_map_img = nib.Nifti1Image(post_map, mask_img.affine)
    fig=plot_stat_map(post_map_img,
            bg_img=nib.Nifti1Image(mni_data, mni_img.affine),
            colorbar=True,
            threshold=0,
            cbar_tick_format="%.3f",
            #vmax=0.003, vmin=-0.003,
            cmap='coolwarm',
            #cmap = base_cmap,
            cut_coords=[0, -22, 13])
    # save plot as jpg # overwrite option




def get_ppc_curve():
    for i in range(ppc.shape[0]):
        density = fastKDE.pdf(ppc[i].reshape(-1),ecf_precision=1) # flat to vector
        density.plot(alpha=0.2, color="#E69F00", label='Posterior predictive' if i == 0 else None)

    data_den= fastKDE.pdf(data.reshape(-1),ecf_precision=1)
    data_den.plot(alpha=1.0, color="black", label='Observed data')
    plt.xlim(-0.04, 0.04)  
    plt.ylabel('')
    plt.xlabel('')

    # Add legend (only one entry for grey lines)
    plt.legend()
    #plt.savefig(f'plots/M21_ppc_L{L}.png', dpi=300, bbox_inches='tight')  
    plt.show()

def get_pyrba_args():

    parser = argp.ArgumentParser(prog = 'pyrba_wrapper.py',
                                formatter_class=argp.RawTextHelpFormatter)

    parser.add_argument('-data_path', "--data_path",type=dir_path)

    parser.add_argument('-mask_path', "--mask_path",type=dir_path)

    #[YNS] : what should be the default model the user should use
    parser.add_argument('-model_name', "--model_name")

    return parser.parse_args()




def run_vi(Y, X, grids, kernel, L, L_eta, dtype, verbose,
           ELBO_diff_tol, para_diff_tol, elbo_stop, max_iter, module_base="model", **extra):
    mod = importlib.import_module(f"{module_base}.BSHE_VI")
    Model = getattr(mod, "BSHE_VI")
    model = Model(
        Y=Y, X=X, grids=grids, kernel=kernel,
        L=L, L_eta=L_eta,
        verbose=verbose, dtype=dtype,
        ELBO_diff_tol=ELBO_diff_tol, para_diff_tol=para_diff_tol,
        elbo_stop=elbo_stop, max_iter=max_iter,
        **extra
    )
    entry = getattr(model, "fit", None) or getattr(model, "run", None) or getattr(model, "optimize", None)
    result = entry()
    if isinstance(result, tuple) and len(result) == 2:
        paras, profile = result
    else:
        paras, profile = result, {}
    return model, paras, profile



def run_gibbs(Y, X, grids, kernel, L, L_eta, dtype,
            burnin, thin, mcmc_sample, module_base="model", **extra):
    mod = importlib.import_module(f"{module_base}.BSHE_Gibbs")
    Model = getattr(mod, "BSHE_Gibbs")
    model = Model(
        Y=Y, X=X, grids= grids, kernel=kernel,
        L=L, L_eta=L_eta,
        dtype=dtype,
        burnin=burnin, thin=thin,
        **extra
    )
    entry = getattr(model, "run", None) or getattr(model, "sample", None) or getattr(model, "fit", None)
    result = entry()
    if isinstance(result, tuple) and len(result) == 2:
        paras, profile = result
    else:
        paras, profile = result, {}
    return model, paras, profile


# ----------------- DISPATCH TABLE -----------------
RUNNERS = {
    "BSHE_VI": run_vi,
    "BSHE_Gibbs": run_gibbs,
}


# ----------------- RUN FUNCTION -----------------
def run_bshe(backend: str, **kwargs):
    """Call the right runner with no if-statements."""
    backend = backend.strip()
    if backend not in RUNNERS:
        raise ValueError(f"Unknown backend {backend}. Choices: {list(RUNNERS)}")
    return RUNNERS[backend](**kwargs)

def main():

    args       = get_pyrba_args()
    data_path  = args.data_path
    mask_path  = args.mask_path
    model_name = args.model_name

    print('data_path =',data_path)
    print('mask_path =',mask_path)
    print('model_name =',model_name)
   
    
    #length_scale
    ls = 0.1
    print('length_scale = ', ls)

    # load  data
    x,y,z = load_data(data_path)

    print('xyz read over')

    
    # load mask data
    mask_data = load_mask_data(mask_path)
    print('mask_data is loaded')

    # load pickled data
    narps = load_pickle('data/NARPS/data.pickle')
    data = narps['data']
    #[YNS]: why is xyz redifned again here 
    x,y,z = narps['coord']
    S = narps['S']
    dtype = torch.float32
    print('pickled data is loaded')

    
    # choose kernel function
    # kernel = RBF(length_scale=ls)
    kernel = Matern(length_scale=ls, nu=1.5)
    #[YNS] : can these be defined as global variables.
    # are they the same values for all the models
    L = 1200 
    L_eta = 50
    n_chain = 3
    print('kernel function is defined')
    #re-import the module in place,
    
    #module = importlib.reload(BSHE_Gibbs) 
    module = import_model_by_name(model_name, package="model")
    print('module imported')
    '''
    model = BSHE_Gibbs.BSHE_Gibbs(Y=data,X=None,
        grids=S, kernel=kernel,  L = L, L_eta = L_eta,
        burnin=5000,
        mcmc_sample=1000,
        thin=2,
        dtype=dtype)
    
    '''
    '''
    #[YNS] : TypeError: get_basis() got an unexpected keyword argument 'err' line 59&60
    model, paras, profile = run_bshe(
    backend="BSHE_Gibbs",
    Y= data, X=None, grids=S, kernel=kernel,
    L= L, L_eta= L_eta, dtype=torch.float32,
    burnin=5000, thin=2,mcmc_sample =1000)

    '''
    model, paras, profile = run_bshe(
    backend="BSHE_VI",
    Y=data, X=None,
    grids=S, kernel=kernel, L = L, L_eta = L_eta,verbose=5000,
    dtype=dtype, ELBO_diff_tol=1e-8,para_diff_tol = 1e-8, elbo_stop=False,max_iter=50000)
    '''
    model = BSHE_VI.BSHE_VI( Y=data, X=None,
    grids=S, kernel=kernel, L = L, L_eta = L_eta,verbose=5000,
    dtype=dtype, ELBO_diff_tol=1e-8,para_diff_tol = 1e-8, elbo_stop=False,max_iter=50000)

     # run model
    paras, profile = model.run()

    voxel_mcmc = paras['E_alpha'][:,None] + paras['E_theta_beta'] @ model.basis.t()

    # get_stat_map
    # get_ppc_curve
    '''



if __name__ == "__main__":
    main()