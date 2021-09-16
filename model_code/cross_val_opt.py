import pystan
import numpy as np
import ugali
from ugali.utils import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import sys
from scipy.stats import truncnorm
from bayes_opt import BayesianOptimization
import pandas as pd
import h5py
import GPy
import GPyOpt
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score
from sklearn.datasets import load_iris, load_boston
from functools import partial

#sm = pystan.StanModel(file='stream_model.stan')

def plot_pretty(dpi=175, fontsize=15, labelsize=15, figsize=(10, 8), tex=True):
    # import pyplot and set some parameters to make plots prettier
    plt.rc('savefig', dpi=dpi)
    plt.rc('text', usetex=tex)
    plt.rc('font', size=fontsize)
    plt.rc('xtick.major', pad=1)
    plt.rc('xtick.minor', pad=1)
    plt.rc('ytick.major', pad=1)
    plt.rc('ytick.minor', pad=1)
    plt.rc('figure', figsize=figsize)
    mpl.rcParams['xtick.labelsize'] = labelsize
    mpl.rcParams['ytick.labelsize'] = labelsize
    mpl.rcParams.update({'figure.autolayout': True})
    
    
def optimize(pp1, pp2, hh, n_int_nodes, n_fi2_nodes, n_width_nodes, n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes):
    print('Running Optimization ...')
    surv = 'atlas_sergey'
    print(surv)
    #n_int_nodes, n_fi2_nodes, n_width_nodes = n[0], n[1],n[2]
    #n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes = n[3], n[4],n[5]
    
    
    n_int_nodes, n_fi2_nodes = np.around(n_int_nodes).astype('int')[0], np.around(n_fi2_nodes).astype('int')[0]
    #n_width_nodes = np.around(n_width_nodes).astype('int')[0]
    n_bg_nodes, n_bgsl_nodes = np.around(n_bg_nodes).astype('int')[0], np.around(n_bgsl_nodes).astype('int')[0]
    n_bgsl2_nodes = np.around(n_bgsl2_nodes).astype('int')[0]
    
    if surv == 'atlas':
        lowphi1, highphi1 = -10.01,10.41
    elif surv == 'atlas_sergey':
        lowphi1, highphi1 = -13.01,10.01
    elif surv == 'phoenix_tall':
        lowphi1, highphi1 = -9.01,7.41
    
    
    #n_int_nodes = int(node_numbers[0])
    int_nodes = np.linspace(lowphi1, highphi1, n_int_nodes)
    print('{} evenly spaced intensity nodes'.format(n_int_nodes))

    #n_fi2_nodes = int(node_numbers[1])
    fi2_nodes = np.linspace(lowphi1, highphi1, n_fi2_nodes)
    print('{} evenly spaced track nodes'.format(n_fi2_nodes))

    #n_width_nodes = int(node_numbers[2])
    width_nodes = np.linspace(lowphi1, highphi1, n_width_nodes)
    print('{} evenly spaced width nodes'.format(n_width_nodes))

    #n_bg_nodes = int(node_numbers[3])
    bg_nodes = np.linspace(lowphi1, highphi1, n_bg_nodes)
    print('{} evenly spaced background nodes'.format(n_bg_nodes))

    #n_bgsl_nodes = int(node_numbers[4])
    bgsl_nodes = np.linspace(lowphi1, highphi1, n_bgsl_nodes)
    print('{} evenly spaced bgsl nodes'.format(n_bgsl_nodes))

    #n_bgsl2_nodes = int(node_numbers[5])
    bgsl2_nodes = np.linspace(lowphi1, highphi1, n_bgsl2_nodes)
    print('{} evenly spaced bgsl2 nodes'.format(n_bgsl2_nodes))

    
    '''
    pp1 = np.load('model_arrays/pp1_full_{}.npy'.format(surv))
    pp2 = np.load('model_arrays/pp2_full_{}.npy'.format(surv))
    vv = np.load('model_arrays/vv_full_{}.npy'.format(surv))
    vv_mask = np.load('model_arrays/vv_mask_full_{}.npy'.format(surv))
    smooth = np.load('model_arrays/smooth_full_{}.npy'.format(surv))
    smooth_res = np.load('model_arrays/smooth_res_full_{}.npy'.format(surv))


    hh = np.ravel(vv_mask).astype(int)
    '''
    idxset = np.arange(len(hh))+1

    datainput = { 
         "n_pix": int(len(hh)),# number of pixels,   
         "hh": hh, 
         "x": pp1, 
         "y": pp2, 
         "n_int_nodes": n_int_nodes,  
         "int_nodes": int_nodes, 
         "n_fi2_nodes": n_fi2_nodes,  
         "fi2_nodes": fi2_nodes, 
         "n_width_nodes": n_width_nodes,  
         "width_nodes": width_nodes, 
         "n_bg_nodes": n_bg_nodes,  
         "bg_nodes": bg_nodes, 
         "n_bgsl_nodes": n_bgsl_nodes,  
         "bgsl_nodes": bgsl_nodes, 
         "n_bgsl2_nodes": n_bgsl2_nodes,  
         "bgsl2_nodes": bgsl2_nodes, 
         "nfit": len(hh[hh>0]), 
         "fitset": idxset[(hh>0)],
         "npred": len(hh[hh==0]),
         "predset": idxset[(hh == 0)],
         "binsizey": 0.05,
         "width_prior_cen": 0.2,
         "lwidth_prior_width": np.log(0.2)}

    pp1_length = len(pp1[0]) - 1

    '''
    if surv == 'atlas':
        ints = smooth_res[90, np.around(pp1_length / (n_int_nodes - 1) * np.linspace(0,n_int_nodes-1,n_int_nodes)).astype('int')]
    elif surv == 'atlas_sergey':
        ints = smooth_res_sergey[90, np.around(pp1_length / (n_int_nodes - 1) * np.linspace(0,n_int_nodes-1, n_int_nodes)).astype('int')]
    elif surv == 'phoenix_tall':
        ints = smooth_res[80, np.around(pp1_length / (n_int_nodes - 1) * np.linspace(0,n_int_nodes-1,n_int_nodes)).astype('int')]
    ints = ints.astype('float64')
    for i in range(len(ints)):
        if ints[i] == 0:
            ints[i] = 0.01
        elif ints[i] < 0:
            ints[i] = 0.01
    log_ints = np.log(ints)
    '''
    
    
    log_widths = np.log(truncnorm.rvs(0.1, 0.4, size = n_width_nodes))
    log_bgs = np.log(np.random.random(n_bg_nodes) * 6 + 4)
    bgsls = np.zeros(n_bgsl_nodes) + np.random.normal(scale = 1)
    bgsls2 = np.zeros(n_bgsl2_nodes) - 2 + np.random.normal(scale = 3)
    if surv == 'atlas':
        fi2s = np.zeros(n_fi2_nodes) + 0.5 + np.random.normal(scale = 0.5) #atlas
    if surv == 'atlas_sergey':
        fi2s = np.zeros(n_fi2_nodes) + 0.5 + np.random.normal(scale = 0.5)
    elif surv == 'phoenix_tall':
        fi2s = np.zeros(n_fi2_nodes) + np.random.normal(scale = 0.2) #phoenix
    
    op = sm.optimizing(data = datainput, init={
                                           'log_widths': log_widths,
                                           'log_bgs': log_bgs,
                                           'bgsls': bgsls, 
                                           'bgsls2': bgsls2,
                                           'fi2s': fi2s},
                      iter = 4000,
                      as_vector = False)
    print(op['value']) # log-likelihood

    return op['value']

filename = "power_spectra/to_kian.h5"
with h5py.File(filename, "r") as f:
    smooth_res_sergey = np.transpose(f['hh'].value)
    smooth_res_sergey = smooth_res_sergey[20:180]
    
pixscale1=0.2
pixscale2=0.05

phi1max, phi1min = 10, -13
phi2max, phi2min = 4, -4
nphi1pix = (phi1max-phi1min)/pixscale1 # for a 0.2 x 0.05 pixel size
nphi2pix = (phi2max-phi2min)/pixscale2 # for a 0.2 x 0.05 pixel size

p1 = np.linspace(-13,10,int(nphi1pix))
p2 = np.linspace(-4,4,int(nphi2pix))
pp1,pp2 = np.meshgrid(p1,p2)

class optimize_gpy:
    def __init__(self):
        self.input_dim = 5
        
    def reshape(x,input_dim):
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x

    def f(self, X, pp1, pp2, hh):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        fs = np.zeros((X.shape[0], 1))
        for n in range(nfold):
            idx = np.array(range(pp1.shape[0]))
            idx_valid = np.logical_and(idx>=pp1.shape[0]/nfold*n, idx<pp1.shape[0]/nfold*(n+1))
            print('idx_valid = ', idx_valid)
            idx_train = np.logical_not(idx_valid)
            
            n_int_nodes, n_fi2_nodes = X[:,0],X[:,1]
            n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes = X[:,2],X[:,3],X[:,4]

            opt = optimize(pp1[idx_valid], pp2[idx_valid], hh[idx_valid], n_int_nodes, n_fi2_nodes, 15, n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes) #here I'm immediately getting a loglik, maybe i should just output the model
            fs[i] = -np.mean(cross_val_score(opt, hh[idx_valid], cv=3))
        return fs
    
    
gpy_optimize = optimize_gpy()
    
space = [{'name': 'var_1', 'type': 'discrete', 'domain': (10,12,14,16,18,20,22,24,26,28,30)},
         {'name': 'var_2', 'type': 'discrete', 'domain': (4,6,8,10,12,14,16,18,20)},
         {'name': 'var_3', 'type': 'discrete', 'domain': (6,8,10,12,14,16,18,20,22,24,26,28,30)},
         {'name': 'var_4', 'type': 'discrete', 'domain': (3,4,6,8,10,12,14,16,18,20)},
         {'name': 'var_5', 'type': 'discrete', 'domain': (3,4,6,8,10,12,14,16,18,20)}] 


gpyopt_optimizer = GPyOpt.methods.BayesianOptimization(f = partial(gpy_optimize.f, pp1, pp2, hh)
                                                       domain = space,
                                                       initial_design_numdata = 200, 
                                                       acquisition_type='LCB')
max_iter = 1500
gpyopt_optimizer.run_optimization(max_iter, verbosity=True, report_file = 'log/atlas_cv_width15.txt')
