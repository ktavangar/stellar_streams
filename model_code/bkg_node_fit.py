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
    
    # the order in which the output is reported in the output file (add n_width_nodes at the end if width nodes are not fixed)
    n_bg_nodes, n_bgsl_nodes, n_bgsl2_nodes = [int(_) for _ in p]
    minx1, maxx1 = kw['minx1'], kw['maxx1']
    #nodes = [np.linspace(minx1, maxx1, _) for _ in p]
    #if kw.get('masks') is not None:
    #    nodes = [_[__] for _, __ in zip(nodes, kw['masks'])]
    #print(locals().keys(),globals().keys())
    scope=locals()
    (bg_nodes, bgsl_nodes, bgsl2_nodes) = [
                     np.linspace(minx1,maxx1,eval('n_%s_nodes'%_,scope)) 
                     for _ in ['bg', 'bgsl','bgsl2']]

#    n_int_nodes = len(int_nodes)
#    n_width_nodes = len(width_nodes)
#    n_fi2_nodes = len(fi2_nodes)
#    n_bg_nodes = len(bg_nodes)
#    n_bgsl_nodes = len(bgsl_nodes)
#    n_bgsl2_nodes = len(bgsl2_nodes)

    data = {}
    data.update(kw['data'])
    print('\n\n\n\n\n',data)
    data.update(
        dict(
            n_bg_nodes=n_bg_nodes,
            n_bgsl_nodes=n_bgsl_nodes,
            n_bgsl2_nodes=n_bgsl2_nodes,
            bg_nodes=bg_nodes,
            bgsl_nodes=bgsl_nodes,
            bgsl2_nodes=bgsl2_nodes,
        ))
    data['fitset'] = np.arange(len(data['hh'])) + 1
    data['nfit'] = len(data['fitset'])
    data['npred'] = 1
    data['predset'] = [1]

    bg0 = np.mean(data['hh'])

    init = dict(log_bgs=np.zeros(n_bg_nodes) + np.log(bg0),
                bgsls=np.zeros(n_bgsl_nodes),
                bgsls2=np.zeros(n_bgsl2_nodes)
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
        print('\n\n\n\n\n\n\n', X)
        accum += float(X['log_lik_pred'])
    return -accum


def optimizer(data, M, kw, report = 'log/stream_cv_node_opt.txt', max_iter=200):
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

    maxn=40 # was 20 for au
    domain = [
        dict(name='x1', type='discrete', domain=range(5, maxn)),
        dict(name='x2', type='discrete', domain=range(5, maxn)),
        dict(name='x3', type='discrete', domain=range(5, maxn))
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
    report_file = args.report
    print(report_file)
    
    stream = 'phoenix_tall'
    if stream not in ['au', 'atlas_sergey', 'atlas', 'phoenix_tall']:
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
        hh = np.load('model_arrays/vv_mask_full_{}.npy'.format(stream))
        hh = hh.astype('int')
        pp1 = np.load('model_arrays/pp1_full_{}.npy'.format(stream))
        pp2 = np.load('model_arrays/pp2_full_{}.npy'.format(stream)) 
    if stream == 'phoenix_tall':
        minx, maxx, miny, maxy = -9, 7.4, -4, 4
        hh = np.load('model_arrays/vv_mask_full_phoenix_tall.npy')
        hh = hh.astype('int')
        pp1 = np.load('model_arrays/pp1_full_{}.npy'.format(stream))
        pp2 = np.load('model_arrays/pp2_full_{}.npy'.format(stream)) 
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

    
    mask = ~(np.abs(pp2)<0.75) # specific to phoenix, ignore for others
    print(mask)
    n_pix = mask.sum().astype('int')
    print(n_pix)
    M = pystan.StanModel(file='bkg_model.stan')

    if stream == 'atlas_sergey':
        params = [28,11,3]
        fi2_offset = 0.75
        minx1, maxx1 = pp1[mask].min() - 0.1, pp1[mask].max() + 0.1
        bg_nodes, bgsl_nodes, bgsl2_nodes = [np.linspace(minx1, maxx1, _) for _ in params]
    if stream == 'atlas':
        params = [28,11,3]
        fi2_offset = 0.75   
        bg_nodes, bgsl_nodes, bgsl2_nodes = [np.linspace(minx-0.1, maxx+0.1, _) for _ in params]
    if stream == 'phoenix_tall':
        params = [15,10,3]
        fi2_offset = 0
        bg_nodes, bgsl_nodes, bgsl2_nodes = [np.linspace(minx-0.1, maxx+0.1, _) for _ in params]
    elif stream == 'au':
        params = [3,6,3]#4,6,5,3,4]
        fi2_offset = 1.75
    

    n_bg_nodes = len(bg_nodes)
    n_bgsl_nodes = len(bgsl_nodes)
    n_bgsl2_nodes = len(bgsl2_nodes)
    fitset = np.arange(mask.sum()) + 1
    predset = [1]

    data = dict(n_pix=n_pix,
                hh=hh[mask],
                x=pp1[mask],
                y=pp2[mask],
                n_bg_nodes=n_bg_nodes,
                n_bgsl_nodes=n_bgsl_nodes,
                n_bgsl2_nodes=n_bgsl2_nodes,
                bg_nodes=bg_nodes,
                bgsl_nodes=bgsl_nodes,
                bgsl2_nodes=bgsl2_nodes,
                fitset=fitset,
                predset=predset,
                nfit=len(fitset),
                npred=len(predset),
                binsizey=0.05
)
    bg0 = np.mean(hh[mask]) 
    
    init = dict(log_bgs=np.zeros(n_bg_nodes) + np.log(bg0),
                bgsls=np.zeros(n_bgsl_nodes),
                bgsls2=np.zeros(n_bgsl2_nodes)
                )

    if stream == 'atlas_sergey':
        kw = dict(minx1=minx1, maxx1=maxx1, xgrid_id=xgrid_id[mask])
    elif stream == 'atlas':
            kw = dict(minx1=minx-0.1, maxx1=maxx+0.1, xgrid_id=xgrid_id[mask])
    else: # use when width is not fixed
        kw = dict(minx1=minx-0.1, maxx1=maxx+0.1, xgrid_id=xgrid_id[mask])
        
        
    opt = optimizer(data, M, kw, report = report_file) 

    #with open('atlas_node_outputs.csv','a') as fd:
    #    fd.write(str(self.X[np.argmin(self.Y),:]).strip('[]'))

    '''
    if True:
        plt.subplot(131)
        MI, MA = [scipy.stats.scoreatpercentile(hh, _) for _ in [5, 95]]
        tvaxis(hh,
               minx,
               maxx,
               miny,
               maxy,
               ytitle=r'$\phi_2$ [deg]',
               vmax=MA,
               vmin=MI,
               smooth=1,
               cmap='gray_r',
               noerase=True,
               xtitle=r'$\phi_1$ [deg]')

        #plt.gca().xaxis.set_major_formatter(plt.NullFormatter());
        plt.subplot(132)
        tvaxis(XMOD,
               minx,
               maxx,
               miny,
               maxy,
               cmap='gray_r',
               noerase=True,
               xtitle=r'$\phi_1$ [deg]',
               vmax=MA,
               vmin=MI)  #,ytitle=r'$\phi_2$ [deg]');
        plt.subplot(133)
        tvaxis(hh - XMOD,
               minx,
               maxx,
               miny,
               maxy,
               vmax=.5 * MA,
               vmin=-.5 * MA,
               cmap='gray_r',
               noerase=True,
               smooth=1,
               xtitle=r'$\phi_1$ [deg]')
        plt.tight_layout()
        #plt.subplots_adjust(wspace=0.05)
        plt.gcf().set_size_inches((3.37 * 2, .8 * 3.37))
        #plt.savefig('decals_map_%s.pdf'%(stream))
 
        exec (idlsave.save('stream_model_%s.psav'%stream,'hh,XMOD,minx1,maxx1,miny,maxy,minx,miny'))

    #RES1 = M.sampling(data=data, init=5*[RES], chains=5,
    #                  pars=['log_ints', 'log_widths', 'log_bgs', 'bgsls', 'fi2s']
    #              )
    #XR=RES1.extract()
    kw = dict(minx1=minx1, maxx1=maxx1, xgrid_id=xgrid_id[mask],
              n_width_nodes = n_width_nodes, fi2_offset=fi2_offset)
    '''