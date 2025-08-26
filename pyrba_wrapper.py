import numpy             as np
import torch
import nibabel           as nib
import matplotlib.pyplot as plt
import importlib
#from   model.utils      import save_pickle, load_pickle
from model.utils import save_pickle, load_pickle
from   nilearn.plotting import plot_stat_map
from   sklearn.gaussian_process.kernels import RBF, Matern
from   fastkde import fastKDE

import model.BSHE_Gibbs as BSHE_Gibbs
import model.BSHE_VI    as BSHE_VI


list_model = [ 'BSHE_Gibbs','BSHE_VI']


'''
usage example 
python pyrba_wrapper.py  
       -data_path  ../data/group_analysis.ttest.1grp.equalRange.gain/MNI152_2009_template_SSW.nii.gz
       -mask_path  ../data/group_analysis.ttest.1grp.equalRange.gain/group_mask.inter.nii.gz
       -module_name BSHE_VI

'''
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def load_data(data_path):





    return x,y,z

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

def main():

    args       = get_pyrba_args()
    data_path  = args.data_path
    mask_path  = args.mask_path
    model_name = args.model_name

    print('data_path =',data_path)
    print('mask_path =',data_path)
    print('model_name =',data_path)
    '''
    # length_scale
    ls = 0.1

    # load  data
    x,y,z = load_data(data_path)

    # load mask data
    mask_data = load_mask_data(mask_path)

    # load pickled data

    narps = load_pickle('data/NARPS/data.pickle')
    data = narps['data']
    x,y,z = narps['coord']
    S = narps['S']
    dtype = torch.float32

    # choose kernel function
    # kernel = RBF(length_scale=ls)
    kernel = Matern(length_scale=ls, nu=1.5)
    #[YNS] : can these be defined as global variables.
    # are they the same values for all the models
    L = 1200 
    L_eta = 50
    n_chain = 3

    #re-import the module in place,
    
    module = importlib.reload(model_name) 
   

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