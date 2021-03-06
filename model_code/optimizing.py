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

sm = pystan.StanModel(file='stream_model.stan')

def plot_pretty(dpi=100, fontsize=5, labelsize=15, figsize=(10, 8), tex=True):
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
    
#plot_pretty(fontsize=10)

def optimize(stream, n_int_nodes, n_fi2_nodes, n_width_nodes, n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes):
    print('Running Optimization ...')
    #stream = 'phoenix_tall'
    print(stream)
    #n_int_nodes, n_fi2_nodes, n_width_nodes = n[0], n[1],n[2]
    #n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes = n[3], n[4],n[5]
    #plot_pretty(fontsize=10)
    
    if stream == 'atlas_sergey':
        minx, maxx, miny, maxy = -13, 10, -5, 5
        filename = "power_spectra/to_kian.h5"
        with h5py.File(filename, "r") as f:
            hh = f['hh'].value
           # hh = 2*hh
            
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
    if stream == 'sim_phoenix':
        minx, maxx, miny, maxy = -9, 7.4, -2, 1.5
        hh = np.load('simulated_phoenix_catalog.npy')
        hh = hh[90:]
        pp1 = np.load('model_arrays/pp1_full_phoenix_tall.npy')
        pp1 = pp1[:,1:-1]
        pp1 = pp1[90:]        
        pp2 = np.load('model_arrays/pp2_full_phoenix_tall.npy') 
        pp2 = pp2[:,1:-1]
        pp2 = pp2[90:]
        pp2 = pp2-2.5
        #hh[np.where(hh==0) & (pp1<-6)] = np.nan
        #hh[np.where(hh>16)] = np.nan
        hh = hh.astype('int')
        vv_mask=hh
    if stream == 'au': 
        minx, maxx, miny, maxy = -21, -10, -5, 5

    mask = hh > -1
    #mask = ~((pp1<5.6) & (pp2<-0.2) & (pp1>4.8) & (pp2>-0.6)) #phoenix
    #((pp1<-6.3) & (pp2<-1.5)) | 
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
    if stream == 'sim_phoenix':
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
                fi2s=np.zeros(n_fi2_nodes)+fi2_offset,
                #fi2s=fi2_offset - .2 * ((fi2_nodes - 7.5) / 10)**2
               )
    
    op = sm.optimizing(data = datainput, init=init,
                      iter = 4000,
                      as_vector = False)
    print(op['value'])
    print('Done')
    print(' ')
    
    
    print('Getting model values ...')
    xmod = op['par']['xmod']
    mod_mean_peak = np.zeros(pp1.shape)
    mod_mean_peak[mask] = xmod.astype('float64')
    model_map_log = np.array(mod_mean_peak).reshape(pp1.shape)
    model_map = np.e**model_map_log
    
    print("Getting Log Intensity means ...")
    logint_val = op['par']['logint_val']
    int1d_mean = np.zeros(pp1.shape)
    int1d_mean[mask] = logint_val.astype('float64')
    int1d_map_log = np.array(int1d_mean).reshape(pp1.shape)
    int1d_map = np.e**int1d_map_log
    if stream == 'atlas_sergey':
        int1d = []
        for i in range(len(int1d_map)):
            int1d = np.append(int1d, np.median(int1d_map[i]))
    

    print('Getting Stream Intensity means ...')
    logint_pix = op['par']['logint_pix']
    logint_mean = np.zeros(pp1.shape)
    logint_mean[mask] = logint_pix.astype('float64')
    int_map_log = np.array(logint_mean).reshape(pp1.shape)
    int_map = np.e**int_map_log
    print(np.sum(int_map))


    print('Getting Background Intensity means ...')
    logbg_pix = op['par']['logbg_pix']   
    bkg_mean = np.zeros(pp1.shape)
    bkg_mean[mask] = logbg_pix.astype('float64')
    bkg_map_log = np.array(bkg_mean).reshape(pp1.shape)
    bkg_map = np.e**bkg_map_log  


    print("Getting Width means ...")
    width_val = op['par']['width_val']
    wid_mean = np.zeros(pp1.shape)
    wid_mean[mask] = width_val.astype('float64')
    width_map = np.array(wid_mean).reshape(pp1.shape)
    if stream == 'atlas_sergey':
        width = []
        for i in range(len(width_map)):
            width = np.append(width, np.median(width_map[i]))
    


    print("Getting Phi 2 means ...")
    fi2_val = op['par']['fi2_val']
    phi2_mean = np.zeros(pp1.shape)
    phi2_mean[mask] = fi2_val.astype('float64')
    fi2_map = np.array(phi2_mean).reshape(pp1.shape)
    if stream == 'atlas_sergey':
        fi2 = []
        for i in range(len(fi2_map)):
            fi2 = np.append(fi2, np.median(fi2_map[i]))
            
    #np.save('model_arrays/atlas_opt_nodes_int1d.npy', int1d_map)        
    #np.save('model_arrays/atlas_opt_nodes_width.npy', width_map) 
    #np.save('model_arrays/atlas_opt_nodes_fi2.npy', fi2_map)
    
    #np.save('power_spectra/pp1_atlas.npy', xgrid)
    #np.save('power_spectra/int1dmap0_atlas.npy', int1d_map[0])
    print("Plotting...")
    fig, axs = plt.subplots(4,2, figsize=(8,8), sharex=True)

    if stream == 'atlas_sergey':
        axs[0][0].pcolormesh(pp1, pp2, hh, vmin = 0, vmax = 5, cmap='gray_r')
        axs[0][0].set_xlim(maxx,minx)
        axs[0][0].set_ylim(-4,4)
        axs[0][0].set_title(r' Stream Data')
    else:
        axs[0][0].pcolormesh(pp1, pp2, vv_mask, vmin = 0, vmax = 8, cmap='gray_r')
        axs[0][0].set_xlim(maxx,minx)
        axs[0][0].set_ylim(-4,4)
        axs[0][0].set_title(r' Stream Data')

    if stream == 'atlas_sergey':
        axs[1][0].pcolormesh(pp1, pp2, model_map, vmin = 0, vmax = 5, cmap='gray_r')
    else:
        axs[1][0].pcolormesh(pp1, pp2, model_map, vmin = 0, vmax = 10, cmap='gray_r')
    axs[1][0].set_xlim(maxx,minx)
    axs[1][0].set_ylim(-4,4)
    axs[1][0].set_title(r' Stream Model')

    if stream == 'atlas_sergey':
        axs[2][0].pcolormesh(pp1, pp2, int_map, vmin = 0, vmax = 5, cmap='gray_r')
    else:
        axs[2][0].pcolormesh(pp1, pp2, int_map, vmin = 0, vmax = 7, cmap='gray_r')
    axs[2][0].set_xlim(maxx,minx)
    axs[2][0].set_ylim(-4,4)
    axs[2][0].set_title(r'Stream Model w/o Background')

    if stream == 'atlas_sergey':
        axs[3][0].pcolormesh(pp1, pp2, bkg_map, vmin = 0,vmax = 5, cmap='gray_r')
    else:
        axs[3][0].pcolormesh(pp1, pp2, bkg_map, vmin = 3,vmax = 10, cmap='gray_r')
    axs[3][0].set_xlim(maxx,minx)
    axs[3][0].set_ylim(-4,4)
    axs[3][0].set_xlabel(r'Phi 1')
    axs[3][0].set_title(r'Background Model')

    if stream == 'atlas_sergey':
        axs[0][1].pcolormesh(pp1, pp2, hh-model_map, vmin = -1.5,vmax = 1.5, cmap='gray_r')
        axs[0][1].set_xlim(maxx,minx)
        axs[0][1].set_ylim(-4,4)
        axs[0][1].set_title(r'Residual')
    else:
        axs[0][1].pcolormesh(pp1, pp2, vv_mask-model_map, vmin = -1.5,vmax = 1.5, cmap='gray_r')
        axs[0][1].set_xlim(maxx,minx)
        axs[0][1].set_ylim(-2,2)
        axs[0][1].set_title(r'Residual')

    
    if stream == 'atlas_sergey':
        axs[1][1].plot(pp1[:,0], int1d, c = 'r')
    elif stream == 'sim_phoenix':
        axs[1][1].plot(pp1[0], int1d_map[0], c = 'r')
    else:
        axs[1][1].plot(pp1[0], int1d_map[80], c = 'r')
    axs[1][1].set_xlim(maxx,minx)
    axs[1][1].set_ylim(0, 120)
    axs[1][1].scatter(np.linspace(minx,maxx,n_int_nodes), np.zeros(n_int_nodes), c = 'k', s = 50, label = 'Intensity')
    axs[1][1].legend()
    axs[1][1].set_title(r'{} nodes'.format(n_int_nodes))

    
    if stream == 'atlas_sergey':
        axs[2][1].plot(pp1[:,0], width, c = 'r')
    elif stream == 'sim_phoenix':
        axs[2][1].plot(pp1[0], width_map[0], c = 'r')
    else:
        axs[2][1].plot(pp1[0], width_map[80], c = 'r')
    axs[2][1].set_xlim(maxx,minx)
    axs[2][1].set_ylim(0, 0.5)
    axs[2][1].scatter(np.linspace(minx,maxx,n_width_nodes), np.zeros(n_width_nodes)+0.05,c = 'k', s = 50, label = 'Width')
    axs[2][1].legend()
    #axs[2][1].set_title(r'Stream Width')

    if stream == 'atlas_sergey':
        axs[3][1].plot(pp1[:,0], fi2, c = 'r')
    elif stream == 'sim_phoenix':
        axs[3][1].plot(pp1[0], fi2_map[0], c = 'r')
    else:
        axs[3][1].plot(pp1[0], fi2_map[80], c = 'r')
    axs[3][1].set_xlim(maxx,minx)
    axs[3][1].set_ylim(-0.5, 1.5)
    axs[3][1].scatter(np.linspace(minx,maxx,n_fi2_nodes), np.zeros(n_fi2_nodes)-0.45,c = 'k', s = 50, label = 'Track')
    axs[3][1].legend()
    
    fig.suptitle('{} Stream'.format(stream), fontsize=20)
    #axs[3][1].set_title(r'Stream Track')
    
    '''
    phi1_mem = np.load('phi1_mem.npy')
    phi1_highmem = np.load('phi1_highmem.npy')
    phi2_mem = np.load('phi2_mem.npy')
    phi2_highmem = np.load('phi2_highmem.npy')
    plt.plot(pp1[0], fi2_map[0], c = 'r')
    plt.plot(pp1[0], fi2_map[0] + width_map[0], c = 'b')
    plt.plot(pp1[0], fi2_map[0] - width_map[0], c = 'b')
    plt.scatter(phi1_mem, phi2_mem)
    plt.scatter(phi1_highmem, phi2_highmem)
    plt.xlim(7.4, -9)
    plt.ylim(-0.5, 0.5)
    plt.xlabel(r'Phi 1')
    plt.title(r'Stream Track and Width')
    '''
    return op['value']

#gpyopt_optimizer.plot_convergence()
#print('Atlas Sergey Data: {}'.format(optimizer.max))

#optimize(17, 8, 15, 29, 9, 7) # (Li et al data) track had more 5s but 5 was the lowest so picked the next best

#optimize('atlas', 14, 12, 15, 14, 9, 3) # optimized for Y6 data


#surv = 'phoenix_tall'
#optimize('phoenix', 22,16,8,16,8,7) # test=3
#optimize('phoenix', 22,16,8,16,8,3) # test=4
#optimize('phoenix', 20,16,8,16,8,3) # test=5
#optimize('phoenix', 20,14,7,15,8,6)
#optimize('phoenix', 9,14,7,15,8,6)

#flat track phoenix
#optimize('phoenix', 20,3,7,15,8,6)

#simulate phoenix
optimize('sim_phoenix', 5, 5, 3, 5, 9, 3)
optimize('sim_phoenix', 3, 3, 3, 5, 9, 3)

plt.show()

