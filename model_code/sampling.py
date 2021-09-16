import numpy as np
import pystan
import pickle
import pystan.misc
from scipy.stats import truncnorm
import h5py

test = 7
#print('Running wider phoenix model')

stanm = pystan.StanModel(file="stream_model.stan") 

def sample(n_int_nodes, n_fi2_nodes, n_width_nodes, n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes):
    print('Running Optimization ...')
    stream = 'phoenix'
    print(stream)
    #n_int_nodes, n_fi2_nodes, n_width_nodes = n[0], n[1],n[2]
    #n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes = n[3], n[4],n[5]
    
    
    if stream == 'atlas_sergey':
        minx, maxx, miny, maxy = -13, 10, -5, 5
        filename = "power_spectra/to_kian.h5"
        with h5py.File(filename, "r") as f:
            hh = f['hh'].value
            hh = hh[20:180]
            
        myrange = [[minx, maxx], [miny, maxy]]
    
        xgrid = np.linspace(myrange[0][0], myrange[0][1], hh.shape[0] + 1,
                            True)
        xgrid = xgrid[:-1] + .5 * (xgrid[1] - xgrid[0])
        ygrid = np.linspace(myrange[1][0], myrange[1][1], hh.shape[1] + 1,
                        True)
        ygrid = ygrid[:-1] + .5 * (ygrid[1] - ygrid[0])
        pp1 = xgrid[:, None] + ygrid[None, :] * 0
        pp2 = xgrid[:, None] * 0 + ygrid[None, :]
       
        
    if stream == 'atlas':
        minx, maxx, miny, maxy = -10, 13, -4, 4
        hh = np.load('atlas_stream_coord_catalog.npy')
        hh = np.transpose(hh)
        pp1 = np.load('model_arrays/pp1_full_{}.npy'.format(stream))
        pp1 = pp1[:,1:-1]
        pp2 = np.load('model_arrays/pp2_full_{}.npy'.format(stream)) 
        pp2 = pp2[:,1:-1]
        hh[np.where((hh==0) & (pp1<-5))] = np.nan
        hh = hh.astype('int')
        vv_mask=hh
    if stream == 'phoenix':
        minx, maxx, miny, maxy = -9, 7.4, -4, 4
        hh = np.load('phoenix_data_catalogs/phoenix_stream_coord_catalog_paper_final.npy')
        hh = np.transpose(hh)
        pp1 = np.load('model_arrays/pp1_full_phoenix_tall.npy')
        pp1 = pp1[:,1:-1]
        pp2 = np.load('model_arrays/pp2_full_phoenix_tall.npy') 
        pp2 = pp2[:,1:-1]
        hh[np.where((hh==0) & (pp1<-6))] = np.nan
        hh[np.where(hh>16)] = np.nan
        hh = hh.astype('int')
        vv_mask=hh
    if stream == 'au':
        minx, maxx, miny, maxy = -21, -10, -5, 5

    mask = hh > -1
    n_pix = mask.sum()
    

    if stream == 'atlas_sergey':
        params = [n_int_nodes, n_fi2_nodes, n_width_nodes, n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes]
        fi2_offset = 0.75
    if stream == 'atlas':
        params = [n_int_nodes, n_fi2_nodes, n_width_nodes, n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes]
        fi2_offset = 0.75        
    if stream == 'phoenix':
        params = [n_int_nodes, n_fi2_nodes, n_width_nodes, n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes]
        fi2_offset = 0
    elif stream == 'au':
        params = [n_int_nodes, n_fi2_nodes, n_width_nodes, n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes]
        fi2_offset = 1.75
    int_nodes, fi2_nodes, width_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes = [np.linspace(minx-0.1, maxx+0.1, _) for _ in params]
    
    
    fitset = np.arange(mask.sum()) + 1
    predset = [1]
    '''
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
    '''

    datainput = dict(n_pix=n_pix,
                hh=hh[mask],
                x=pp1[mask],
                y=pp2[mask],
                n_int_nodes=n_int_nodes,
                n_width_nodes=n_width_nodes,
                n_fi2_nodes=n_fi2_nodes,
                n_bg_nodes=n_bg_nodes,
                n_bgsl_nodes=n_bgsl_nodes,
                n_bgsl2_nodes=n_bgsl2_nodes,
                int_nodes=int_nodes,
                width_nodes=width_nodes,
                fi2_nodes=fi2_nodes,
                bg_nodes=bg_nodes,
                bgsl_nodes=bgsl_nodes,
                bgsl2_nodes=bgsl2_nodes,
                fitset=fitset,
                predset=predset,
                nfit=len(fitset),
                npred=len(predset),
                binsizey=0.05,
                width_prior_cen=0.2,
                lwidth_prior_width=0.5)
    bg0 = np.mean(hh[mask]) 
    
    w0=0.2
    init = dict(log_bgs=np.zeros(n_bg_nodes) + np.log(bg0),
                log_widths=np.zeros(n_width_nodes) + np.log(w0),
                log_ints=np.zeros(n_int_nodes) + np.log(bg0),
                bgsls=np.zeros(n_bgsl_nodes),
                bgsls2=np.zeros(n_bgsl2_nodes),
                fi2s=np.zeros(n_fi2_nodes),
                #fi2s=fi2_offset - .2 * ((fi2_nodes - 7.5) / 10)**2
               )

    print("begin run")
    n_iter = 5000
    fit_model = stanm.sampling(data=datainput, init = 12*[init], iter=n_iter, chains=12, warmup=2000,
                           control = dict(max_treedepth=15, adapt_delta=0.95))

    print("run finished")
    with open('phoenix_models/{}_model_fit_final_test{}.pickle'.format(stream, test), 'wb') as f:
        pickle.dump({'model': stanm, 'fit':fit_model}, f, protocol=-1)
    print('all done')
    
#sample(20,12,10,16,8,3) #test=1
#sample(22,16,8,16,8,7) # test=3
#sample(22,16,8,16,8,3) # test=4
#sample(20,16,8,16,8,3) # test=5
#sample(20,14,7,15,8,6) #test=6
sample(21,14,8,17,10,6) #test=7
