""" 
Author: Kiyan Tavangar (2021) 
Adapted from: Sergey Koposov (CMU/University of Edinburgh)
Email: ktavangar@uchicago.edu

"""

import numpy as np
import scipy.interpolate
import pystan
import matplotlib.pyplot as plt
import copy
import GPyOpt
import h5py
from scipy.stats import truncnorm
import pickle

def get_cvid(N, ncv):
    st=np.random.get_state()
    np.random.seed(10012)
    cvid = np.random.permutation(N)%ncv
    np.random.set_state(st)
    return cvid

def betw(x, x1, x2):
    return (x >= x1) & (x < x2)


def funcer(p, **kw):
    ncv = 3
    #n_width_nodes = kw['n_width_nodes']
    #n_bgsl2_nodes = kw['n_bgsl2_nodes']
    #n_fi2_nodes = kw['n_fi2_nodes']
    #n_bg_nodes = kw['n_bg_nodes']
    #n_bgsl_nodes = kw['n_bgsl_nodes']# flat track experiment
    fi2_offset = kw['fi2_offset']
    
    # the order in which the output is reported in the output file (add n_width_nodes at the end if width nodes are not fixed)
    n_fi2_nodes, n_int_nodes, n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes, n_width_nodes = [int(_) for _ in p]
    #n_int_nodes = int(p[0]) #just intensity experiment
    minx1, maxx1 = kw['minx1'], kw['maxx1']
    #nodes = [np.linspace(minx1, maxx1, _) for _ in p]
    #if kw.get('masks') is not None:
    #    nodes = [_[__] for _, __ in zip(nodes, kw['masks'])]
    #print(locals().keys(),globals().keys())
    scope=locals()
    (fi2_nodes, int_nodes, bg_nodes, bgsl_nodes,
     bgsl2_nodes, width_nodes) = [
                     np.linspace(minx1,maxx1,eval('n_%s_nodes'%_,scope)) 
                     for _ in ['fi2', 'int', 'bg', 'bgsl','bgsl2', 'width']]

#    n_int_nodes = len(int_nodes)
#    n_width_nodes = len(width_nodes)
#    n_fi2_nodes = len(fi2_nodes)
#    n_bg_nodes = len(bg_nodes)
#    n_bgsl_nodes = len(bgsl_nodes)
#    n_bgsl2_nodes = len(bgsl2_nodes)

    data = {}
    data.update(kw['data'])
    data.update(
        dict(
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
        ))
    data['fitset'] = np.arange(len(data['hh'])) + 1
    data['nfit'] = len(data['fitset'])
    data['npred'] = 1
    data['predset'] = [1]

    bg0 = np.mean(data['hh'])

    w0 = 0.2
    init = dict(log_bgs=np.zeros(n_bg_nodes) + np.log(bg0),
                log_widths=np.zeros(n_width_nodes) + np.log(w0),
                log_ints=np.zeros(n_int_nodes) + np.log(bg0),
                bgsls=np.zeros(n_bgsl_nodes),
                bgsls2=np.zeros(n_bgsl2_nodes),
                #fi2s=fi2_offset - .2 * ((fi2_nodes - 7.5) / 10)**2 #for atlas
                fi2s=np.zeros(n_fi2_nodes)

                #            (fi2_nodes-64)*0.1
                #np.zeros(n_fi2_nodes)
                )

    RES = kw['M'].optimizing(data=data, init=init)
    #fitset = np.random.permutation(len(data['hh']))
    #fitset = get_cvid(len(data['hh']),ncv)
    fitset = kw['xgrid_id']%ncv
    #1/0
    accum = 0
    for i in range(ncv):
        fitset_cur = fitset != i
        data['fitset'] = np.nonzero(fitset_cur)[0] + 1
        data['predset'] = np.nonzero(~fitset_cur)[0] + 1
        data['nfit'] = len(data['fitset'])
        data['npred'] = len(data['predset'])
        X = kw['M'].optimizing(data=data, init=RES)
        accum += float(X['log_lik_pred'])
    return -accum


def optimizer(data, M, kw, report = 'log/stream_cv_node_opt.txt', max_iter=75):
    newkw = copy.copy(kw)
    newkw['data'] = data
    newkw['M'] = M

    cache={}
    def GG(x):
        ret = np.zeros(len(x))
        for i, curx in enumerate(x):
            print(i, curx)
            if tuple(curx) not in cache:
                ret[i] = funcer(curx, **newkw)
            cache[tuple(curx)] = ret[i]
        print('ret = {}'.format(ret))
        return ret

    domain = [
        dict(name='x1', type='discrete', domain=range(5, 30)),
        dict(name='x2', type='discrete', domain=range(5, 30)),
        dict(name='x3', type='discrete', domain=range(5, 30)),
        dict(name='x4', type='discrete', domain=range(3, 20)),
        dict(name='x5', type='discrete', domain=range(3, 10)),
        dict(name='x6', type='discrete', domain=range(3, 30)) # when width is not fixed
    ]

    opt = GPyOpt.methods.BayesianOptimization(
        f=GG,
        domain=domain,
        exact_feval=True,
        num_cores=24,
        batch_size=24,
        initial_design_numdata=24,
        evaluator_type='sequential',
        verbosity=True)
#    max_iter = 20
    opt.run_optimization(max_iter, verbosity=True, report_file = report, eps=0)
    return opt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r','--report',default=None)
    args = parser.parse_args()
    report_file_base = args.report
    print(report_file_base)
    
    stream = 'phoenix'
    if stream not in ['au', 'atlas_sergey', 'atlas', 'phoenix', 'sim_phoenix', 'flat_track_phoenix']:
        raise Exception('not known')
    plt.rc('font', **{
        'size': 7,
        'sans-serif': ['Helvetica'],
        'family': 'sans-serif'
    })
    plt.rc('legend', **{'fontsize': 7})
    plt.rc(
        "text.latex",
        preamble=[
            "\\usepackage{helvet}\\usepackage[T1]{fontenc}\\usepackage{sfmath}"
        ])
    plt.rc("text", usetex=True)
    plt.rc('ps', usedistiller='xpdf')
    plt.rc('savefig', **{'dpi': 300})

    if stream == 'atlas_sergey':
        minx, maxx, miny, maxy = -13, 10, -5, 5
        filename = "power_spectra/to_kian.h5"
        with h5py.File(filename, "r") as f:
            hh = f['hh'].value
        hh = hh.astype('int')
    if stream == 'atlas':
        minx, maxx, miny, maxy = -10, 13, -4, 4
        hh = np.load('phoenix_data_catalogs/atlas_stream_coord_catalog.npy')
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
        minx, maxx, miny, maxy = -9, 7.4, 0.5, 4
        hh = np.load('phoenix_data_catalogs/simulated_phoenix_catalog.npy')
        hh = hh[90:]
        pp1 = np.load('model_arrays/pp1_full_phoenix_tall.npy')
        pp1 = pp1[:,1:]
        pp1 = pp1[90:]
        pp2 = np.load('model_arrays/pp2_full_phoenix_tall.npy') 
        pp2 = pp2[:,1:]
        pp2 = pp2[90:]
        pp2 = pp2-2.5
        #hh[np.where(hh==0) & (pp1<-6)] = np.nan
        hh[np.where(hh>16)] = np.nan
        hh = hh.astype('int')
        vv_mask=hh
    if stream == 'flat_track_phoenix':
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
    
    myrange = [[minx, maxx], [miny, maxy]]
    
    xgrid = np.linspace(myrange[0][0], myrange[0][1], hh.shape[0] + 1,
                        True)
    xgrid = xgrid[:-1] + .5 * (xgrid[1] - xgrid[0])
    if stream == 'atlas_sergey':
        ygrid = np.linspace(myrange[1][0], myrange[1][1], hh.shape[1] + 1,
                        True)
        ygrid = ygrid[:-1] + .5 * (ygrid[1] - ygrid[0])
        pp1 = xgrid[:, None] + ygrid[None, :] * 0
        pp2 = xgrid[:, None] * 0 + ygrid[None, :]
        xgrid_id = np.arange(xgrid.shape[0])[:, None]+ pp1*0
    else:
        xgrid_id = np.arange(xgrid.shape[0])[:, None]+ pp1*0

    M = pystan.StanModel(file='stream_model.stan')
    mask = hh > -1 # I use this for no mask, just so i don't have to change mask-related code later
    #mask = ~((pp1>3.5) & (pp1<4.5) & (pp2>-2.8) & (pp2<-1.8)) # specific to ATLAS, ignore for others
    n_pix = mask.sum()
 
    if stream == 'atlas_sergey':
        params = [10,28,17,11,3, 15]
        fi2_offset = 0.75
        minx1, maxx1 = pp1[mask].min() - 0.1, pp1[mask].max() + 0.1
        fi2_nodes, int_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes, width_nodes = [np.linspace(minx1, maxx1, _) for _ in params]
    if stream == 'atlas':
        params = [10,17,28,11,3, 15]
        fi2_offset = 0.75   
        fi2_nodes, int_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes, width_nodes = [np.linspace(minx-0.1, maxx+0.1, _) for _ in params]
    if stream == 'phoenix':
        params = [16,22,18,10,3,8]
        fi2_offset = 0
        fi2_nodes, int_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes, width_nodes = [np.linspace(minx-0.1, maxx+0.1, _) for _ in params]
    if stream == 'sim_phoenix':
        params = [5,5,5,5,5,5]
        fi2_offset = 0
        fi2_nodes, int_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes, width_nodes = [np.linspace(minx-0.1, maxx+0.1, _) for _ in params]
    if stream == 'flat_track_phoenix':
        params = [3,22,15,8,6,8]
        fi2_offset = 0
        fi2_nodes, int_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes, width_nodes = [np.linspace(minx-0.1, maxx+0.1, _) for _ in params]
    elif stream == 'au':
        params = [4,5,3,5,6,3]#4,6,5,3,4]
        fi2_offset = 1.75
        fi2_nodes, int_nodes, bg_nodes, bgsl_nodes, bgsl2_nodes, width_nodes = [np.linspace(minx-0.1, maxx+0.1, _) for _ in params]
    
    
    n_int_nodes = len(int_nodes)
    n_width_nodes = len(width_nodes)
    n_fi2_nodes = len(fi2_nodes)
    n_bg_nodes = len(bg_nodes)
    n_bgsl_nodes = len(bgsl_nodes)
    n_bgsl2_nodes = len(bgsl2_nodes)
    fitset = np.arange(mask.sum()) + 1
    predset = [1]

    data = dict(n_pix=n_pix,
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
                lwidth_prior_width=0.3)
    bg0 = np.mean(hh[mask]) 
    
    init = dict(log_bgs=np.zeros(n_bg_nodes) + np.log(bg0),
                log_widths= np.log(truncnorm.rvs(0.1, 0.4, size = n_width_nodes)),
                log_ints=np.zeros(n_int_nodes) + np.log(bg0),
                bgsls=np.zeros(n_bgsl_nodes),
                bgsls2=np.zeros(n_bgsl2_nodes),
                fi2s=np.zeros(n_fi2_nodes)+fi2_offset
                #fi2s=fi2_offset - .2 * ((fi2_nodes - 7.5) / 10)**2 # for atlas

                #            (fi2_nodes-64)*0.1
                #np.zeros(n_fi2_nodes)
                )

    #RES = M.optimizing(data=data, init=init)
    
    #kw = dict(minx1=minx1, maxx1=maxx1, xgrid_id=xgrid_id[mask],
    #      n_width_nodes = n_width_nodes, fi2_offset=fi2_offset)
    if stream == 'atlas_sergey':
        kw = dict(minx1=minx1, maxx1=maxx1, xgrid_id=xgrid_id[mask], n_width_nodes = n_width_nodes, fi2_offset=fi2_offset)
    elif stream == 'atlas':
        kw = dict(minx1=minx-0.1, maxx1=maxx+0.1, xgrid_id=xgrid_id[mask], n_width_nodes = n_width_nodes, fi2_offset=fi2_offset)
    elif stream == 'flat_track_phoenix':
        kw = dict(minx1=minx-0.1, maxx1=maxx+0.1, xgrid_id=xgrid_id[mask], n_fi2_nodes = 3, fi2_offset=fi2_offset)
    else: # use when width is not fixed
        kw = dict(minx1=minx-0.1, maxx1=maxx+0.1, xgrid_id=xgrid_id[mask], fi2_offset=fi2_offset)
        
    
    fdict={}
    for i in range(15):
        opath=report_file_base+str(i)+".txt"
        print("starting optimize") 
        opt = optimizer(data, M, kw, report=opath) 
        print("saving data")
        fdict["x"+str(i)]=opt.X
        fdict["Y"+str(i)]=opt.Y
        fdict["best"+str(i)]=opt.x_opt
        with open(report_file_base+"results.pkl", 'wb') as f:
            pickle.dump(fdict, f)
    print("done")
