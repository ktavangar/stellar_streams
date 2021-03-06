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

sm = pystan.StanModel(file='stream_model.stan')

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
    
    
def optimize(int_nodes, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes):
    '''
    Inputs: should be arrays containing the nodes
    '''
    print('Running Optimization ...')
    
    n_int_nodes = len(int_nodes)
    n_fi2_nodes = len(fi2_nodes)
    n_width_nodes = len(width_nodes)
    n_bg_nodes = len(bg_nodes)
    n_bgsl_nodes = len(bgsl_nodes)
    n_bgsl2_nodes = len(bgsl2_nodes)
    
    surv = 'phoenix_tall'
    
    if surv == 'atlas':
        lowphi1, highphi1 = -10.01,10.41
    elif surv == 'phoenix_tall':
        lowphi1, highphi1 = -9.01,7.41

    pp1 = np.load('model_arrays/pp1_full_{}.npy'.format(surv))
    pp2 = np.load('model_arrays/pp2_full_{}.npy'.format(surv))
    vv = np.load('model_arrays/vv_full_{}.npy'.format(surv))
    vv_mask = np.load('model_arrays/vv_mask_full_{}.npy'.format(surv))
    smooth = np.load('model_arrays/smooth_full_{}.npy'.format(surv))
    smooth_res = np.load('model_arrays/smooth_res_full_{}.npy'.format(surv))


    hh = np.ravel(vv_mask).astype(int)
    idxset = np.arange(len(hh))+1

    datainput = { 
         "n_pix": int(len(hh)),# number of pixels,   
         "hh": hh, 
         "x": np.ravel(pp1), 
         "y": np.ravel(pp2), 
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
    
    #find the value of the smoothed data at the intensity nodes
    
    log_ints = np.log(np.random.random(n_int_nodes)*5 + 0.5) #this is potentially not good enough
    log_widths = np.log(truncnorm.rvs(0.1, 0.4, size = n_width_nodes))
    log_bgs = np.log(np.random.random(n_bg_nodes) * 6 + 4)
    bgsls = np.zeros(n_bgsl_nodes) + np.random.normal(scale = 1)
    bgsls2 = np.zeros(n_bgsl2_nodes) - 2 + np.random.normal(scale = 3)
    if surv == 'atlas':
        fi2s = np.zeros(n_fi2_nodes) + 0.5 + np.random.normal(scale = 0.5) #atlas
    elif surv == 'phoenix_tall':
        fi2s = np.zeros(n_fi2_nodes) + np.random.normal(scale = 0.2) #phoenix
    
    op = sm.optimizing(data = datainput, init={'log_ints': log_ints,
                                           'log_widths': log_widths,
                                           'log_bgs': log_bgs,
                                           'bgsls': bgsls, 
                                           'bgsls2': bgsls2,
                                           'fi2s': fi2s},
                      iter = 4000,
                      as_vector = False)
    print(op['value'])
    AIC_log_prob = 2*(n_int_nodes+n_width_nodes+n_fi2_nodes+n_bg_nodes+n_bgsl_nodes+n_bgsl2_nodes) - 2*op['value']
    
    return AIC_log_prob

def iterative(int_nodes, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes):
    
    orig_array = {'intensity': int_nodes, 
        'track': fi2_nodes,
        'width': width_nodes, 
        'bkg': bg_nodes,
        'bkgsl': bgsl_nodes,
        'bkgsl2': bgsl2_nodes}
    
    orig_AIC = optimize(int_nodes, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes)
    best_AIC = orig_AIC
    
    # adding one node at a time
    for i in range(len(int_nodes)-1):
        mid = np.mean([int_nodes[i], int_nodes[i+1]])
        int_nodes_add = np.sort(np.append(int_nodes, np.round(mid, 2)))
        new_AIC = optimize(int_nodes_add, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes)
        if new_AIC < best_AIC:
            best_AIC = new_AIC
            new_array = orig_array.copy()
            new_array['intensity'] = int_nodes_add
            
    for i in range(len(fi2_nodes)-1):
        mid = np.mean([fi2_nodes[i], fi2_nodes[i+1]])
        fi2_nodes_add = np.sort(np.append(fi2_nodes, np.round(mid, 2)))
        new_AIC = optimize(int_nodes, fi2_nodes_add, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes)
        if new_AIC < best_AIC:
            best_AIC = new_AIC
            new_array = orig_array.copy()
            new_array['track'] = fi2_nodes_add

    for i in range(len(width_nodes)-1):
        mid = np.mean([width_nodes[i], width_nodes[i+1]])
        width_nodes_add = np.sort(np.append(width_nodes, np.round(mid, 2)))
        new_AIC = optimize(int_nodes, fi2_nodes, width_nodes_add, bg_nodes, bgsl_nodes, bgsl2_nodes)
        if new_AIC < best_AIC:
            best_AIC = new_AIC
            new_array = orig_array.copy()
            new_array['width'] = width_nodes_add
        
    for i in range(len(bg_nodes)-1):
        mid = np.mean([bg_nodes[i], bg_nodes[i+1]])
        bg_nodes_add = np.sort(np.append(bg_nodes, np.round(mid, 2)))
        new_AIC = optimize(int_nodes, fi2_nodes, width_nodes, bg_nodes_add, bgsl_nodes, bgsl2_nodes)
        if new_AIC < best_AIC:
            best_AIC = new_AIC
            new_array = orig_array.copy()
            new_array['bkg'] = bg_nodes_add
        
    for i in range(len(bgsl_nodes)-1):
        mid = np.mean([bgsl_nodes[i], bgsl_nodes[i+1]])
        bgsl_nodes_add = np.sort(np.append(bgsl_nodes, np.round(mid, 2)))
        new_AIC = optimize(int_nodes, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes_add, bgsl2_nodes)
        if new_AIC < best_AIC:
            best_AIC = new_AIC
            new_array = orig_array.copy()
            new_array['bkgsl'] = bgsl_nodes_add
        
    for i in range(len(bgsl2_nodes)-1):
        mid = np.mean([bgsl2_nodes[i], bgsl2_nodes[i+1]])
        bgsl2_nodes_add = np.sort(np.append(bgsl2_nodes, np.round(mid, 2)))
        new_AIC = optimize(int_nodes, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes_add)
        if new_AIC < best_AIC:
            best_AIC = new_AIC
            new_array = orig_array.copy()
            new_array['bkgsl2'] = bgsl2_nodes_add
    
    if best_AIC == orig_AIC:
        new_array = orig_array
    
    return best_AIC, orig_AIC, new_array
 
    

def iterative2(orig_array, adding_dict):
    
    int_nodes, fi2_nodes, width_nodes = orig_array['intensity'], orig_array['track'], orig_array['width']
    bg_nodes, bgsl_nodes, bgsl2_nodes = orig_array['bkg'], orig_array['bkgsl'], orig_array['bkgsl2']
    orig_AIC = optimize(int_nodes, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes)
    best_AIC = orig_AIC
    
    AIC_prog = []
    # adding one node at a time
    dict_int_add = adding_dict['intensity']
    for i in range(len(dict_int_add)):
        int_nodes_add = np.sort(np.append(orig_array['intensity'], dict_int_add[-i-1]))
        print(int_nodes_add)
        new_AIC = optimize(int_nodes_add, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes)
        orig_array['intensity'] = int_nodes_add
        AIC_prog = np.append(AIC_prog, new_AIC)
        
    dict_fi2_add = adding_dict['track']
    for i in range(len(adding_dict['track'])):
        fi2_nodes_add = np.sort(np.append(orig_array['track'], dict_fi2_add[-i-1]))
        print(fi2_nodes_add)
        new_AIC = optimize(orig_array['intensity'], fi2_nodes_add, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes)
        orig_array['track'] = fi2_nodes_add
        AIC_prog = np.append(AIC_prog, new_AIC)
        
    dict_width_add = adding_dict['width']
    for i in range(len(adding_dict['width'])):
        width_nodes_add = np.sort(np.append(orig_array['width'], dict_width_add[-i-1]))
        new_AIC = optimize(orig_array['intensity'], orig_array['track'], width_nodes_add, bg_nodes, bgsl_nodes, bgsl2_nodes)
        orig_array['width'] = width_nodes_add
        AIC_prog = np.append(AIC_prog, new_AIC)
        
    dict_bg_add = adding_dict['bkg']
    for i in range(len(adding_dict['bkg'])):
        bg_nodes_add = np.sort(np.append(orig_array['bkg'], dict_bg_add[-i-1]))
        new_AIC = optimize(orig_array['intensity'], orig_array['track'], orig_array['width'], bg_nodes_add, bgsl_nodes, bgsl2_nodes)
        orig_array['bkg'] = bg_nodes_add
        AIC_prog = np.append(AIC_prog, new_AIC)
        
    dict_bgsl_add = adding_dict['bkgsl']
    for i in range(len(adding_dict['bkgsl'])):
        bgsl_nodes_add = np.sort(np.append(orig_array['bkgsl'], dict_bgsl_add[-i-1]))
        new_AIC = optimize(orig_array['intensity'], orig_array['track'], orig_array['width'], orig_array['bkg'], bgsl_nodes_add, bgsl2_nodes)
        orig_array['bkgsl'] = bgsl_nodes_add
        AIC_prog = np.append(AIC_prog, new_AIC)
        
    dict_bgsl2_add = adding_dict['bkgsl2']
    for i in range(len(adding_dict['bkgsl2'])):
        bgsl2_nodes_add = np.sort(np.append(orig_array['bkgsl2'], dict_bgsl2_add[-i-1]))
        new_AIC = optimize(orig_array['intensity'], orig_array['track'], orig_array['width'], orig_array['bkg'], orig_array['bkgsl'], bgsl2_nodes_add)
        orig_array['bkgsl2'] = bgsl2_nodes_add
        AIC_prog = np.append(AIC_prog, new_AIC)
    
    return(AIC_prog)
    
       
surv = 'phoenix'
if surv == 'atlas':
    lowphi1, highphi1 = -10.01,10.41
elif surv == 'phoenix':
    lowphi1, highphi1 = -9.01,7.41
    
    
#n_int_nodes = int(node_numbers[0])
total_int_nodes = np.linspace(lowphi1, highphi1, 24)
start_int_nodes = np.take(total_int_nodes, [0,4,8,12,16,20,23])
int_add_nodes = np.take(total_int_nodes, [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22])

total_fi2_nodes = np.linspace(lowphi1, highphi1, 5)
start_fi2_nodes = np.take(total_fi2_nodes, [0,2,4])
fi2_add_nodes = np.take(total_fi2_nodes, [1,3])

total_width_nodes = np.linspace(lowphi1, highphi1, 5)
start_width_nodes = np.take(total_width_nodes, [0,2,4])
width_add_nodes = np.take(total_width_nodes, [1,3])

total_bg_nodes = np.linspace(lowphi1, highphi1, 21)
start_bg_nodes = np.take(total_bg_nodes, [0,4,8,12,16,20])
bg_add_nodes = np.take(total_bg_nodes, [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19])

total_bgsl_nodes = np.linspace(lowphi1, highphi1, 14)
start_bgsl_nodes = np.take(total_bgsl_nodes, [0,3,6,9,11,13])
bgsl_add_nodes = np.take(total_bgsl_nodes, [1,2,4,5,7,8,10,12])

total_bgsl2_nodes = np.linspace(lowphi1, highphi1, 6)
start_bgsl2_nodes = np.take(total_bgsl2_nodes, [0,2,5])
bgsl2_add_nodes = np.take(total_bgsl2_nodes, [1,3,4])
    
    
orig_array = {'intensity': start_int_nodes, 
        'track': start_fi2_nodes,
        'width': start_width_nodes, 
        'bkg': start_bg_nodes,
        'bkgsl': start_bgsl_nodes,
        'bkgsl2': start_bgsl2_nodes}
    
adding_dict = {'intensity': int_add_nodes, 
    'track': fi2_add_nodes,
    'width': width_add_nodes, 
    'bkg': bg_add_nodes,
    'bkgsl': bgsl_add_nodes,
    'bkgsl2': bgsl2_add_nodes}

AIC_prog = iterative2(orig_array, adding_dict)
np.save('AIC_prog_down.npy', AIC_prog)
plt.plot(AIC_prog)  




int_nodes = np.linspace(lowphi1, highphi1, 10)

fi2_nodes = np.linspace(lowphi1, highphi1, 5)

width_nodes = np.linspace(lowphi1, highphi1, 5)

bg_nodes = np.linspace(lowphi1, highphi1, 5)

bgsl_nodes = np.linspace(lowphi1, highphi1, 5)

bgsl2_nodes = np.linspace(lowphi1, highphi1, 5)

'''
new_best_AIC, old_best_AIC, all_nodes = iterative(int_nodes, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes)
while new_best_AIC < old_best_AIC:
    print(' ')
    print(' ')
    print(' ')
    print(' ')
    print(' ')
    print(' ')
    print('Adding new nodes...')
    new_best_AIC, old_best_AIC, allnew_nodes = iterative(all_nodes['intensity'], all_nodes['track'], all_nodes['width'],
                                                      all_nodes['bkg'], all_nodes['bkgsl'], all_nodes['bkgsl2'])
    all_nodes = allnew_nodes
    print(all_nodes)
    
print(new_best_AIC, old_best_AIC)
print(all_nodes)
'''        
