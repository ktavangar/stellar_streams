import numpy as np
import pystan
import pickle
import pystan.misc

print('Running model for ATLAS stream')
n_int_nodes = 17
int_nodes = np.linspace(-13.01, 10.01, n_int_nodes)
#int_nodes = [-9.01, -7.9, -5.9, -4, -3.4, -2.3, -1.6, 0.3, 1.7, 2.1, 3.1, 4.6, 6.3, 7.51]
print(int_nodes)
print('Intensity nodes modified from 17 evenly spaced nodes to reflect visible features')

n_fi2_nodes = 10
fi2_nodes = np.linspace(-13.01, 10.01, n_fi2_nodes)
#fi2_nodes = [-9.01, -7, -3.7,  0, 1.8, 5, 7.51]
#print('7 chosen track nodes')
print('10 evenly spaced track nodes')

n_width_nodes = 15
width_nodes = np.linspace(-13.01, 10.01, n_width_nodes)
#width_nodes = [-9.01, -7, -3.5, 0, 3, 7.51]
#print('6 chosen nodes')
print('15 evenly spaced width nodes')

n_bg_nodes = 28
bg_nodes = np.linspace(-13.01, 10.01, n_bg_nodes)

n_bgsl_nodes = 11
bgsl_nodes = np.linspace(-13.01, 10.01, n_bgsl_nodes)

n_bgsl2_nodes = 3
bgsl2_nodes = np.linspace(-13.01, 10.01, n_bgsl2_nodes)

pp1 = np.load('model_arrays/pp1_atlas.npy')
pp2 = np.load('model_arrays/pp2_atlas.npy')
vv = np.load('model_arrays/vv_atlas.npy')
vv_mask = np.load('model_arrays/vv_mask_atlas.npy')

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
     "binsizey": 0.1,
     "width_prior_cen": 0.25,
     "lwidth_prior_width": np.log(0.2)}

stanm = pystan.StanModel(file="stream_model.stan") 

print("begin run")
fit_model = stanm.sampling(data=datainput, iter=1500, chains=4, warmup=500,
                           control = dict(max_treedepth=14, adapt_delta=0.95))
print("run finished")
with open('atlas_models/atlas_model_fit_test01.pickle', 'wb') as f:
    pickle.dump({'model': stanm, 'fit':fit_model}, f, protocol=-1)
print('all done')
