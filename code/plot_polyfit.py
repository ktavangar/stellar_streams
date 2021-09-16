#!/usr/bin/env python
"""
This is a sandbox for plotting field of streams isochrone selections.
"""
__author__ = "Alex Drlica-Wagner"

import os, sys
import subprocess
from collections import OrderedDict as odict

import matplotlib as mpl
import __main__ as main
interactive = hasattr(main, '__file__')
if not interactive: mpl.use('Agg')

import fitsio
import healpy as hp
import numpy as np
import pylab as plt
from matplotlib.colors import LogNorm

import skymap.survey
from skymap.utils import gal2cel, cel2gal
from skymap.utils import setdefaults
from ugali.utils.shell import mkdir
from skymap.healpix import ang2disc

from numpy.polynomial import polynomial
from polyfit2d import polyfit2d
from streams import GLOBULARS
import streamlib; reload(streamlib)
from streamlib import make_mask, prepare_data, fit_bkg_poly, fit_bkg_gauss
from streamlib import fit_data_bkg, skymap_factory
from streamlib import DATA,CBAR_KWARGS

# Nicer plots
#plt.rc('savefig', dpi=400)
#plt.rc('text', usetex=True)
plt.rc('font', size=15)
plt.rc('xtick.major', pad=5)
plt.rc('xtick.minor', pad=5)
plt.rc('ytick.major', pad=5)
plt.rc('ytick.minor', pad=5)
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
#mpl.rcParams.update({'figure.autolayout': True})


SIGMA = 0.15
FRACMIN = 0.5
NSIDE = 256

def do_plot(arguments):
    mod = arguments
    #if (args.mod is not None) and (i != np.abs(modulus-args.mod).argmin()):
    #    print("Unmatched modulus")
    #    return
    if not np.isclose(mod,modulus).sum():
        print("Unmatched modulus")
        return
    if mod > 20.0:
        print("Modulus greater than 20")
        return
    i = np.abs(modulus-mod).argmin()

    print("%i : m-M = %.1f"%(i,mod))

    hpxmap0 = np.copy(hpxcube[:,i])
    hpxmap = np.copy(hpxcube[:,i])
    #data,bkg = fit_data_bkg(hpxmap,fracdet,fracmin=FRACMIN,sigma=SIGMA)
    
    hpxmap0[hpxmap0 == 0] = np.nan
    cut = np.nanpercentile(hpxmap0, 95)
   
    hpxmap[hpxmap>cut] = 0
    high = np.nanpercentile(hpxmap0, 95)
    low = np.nanpercentile(hpxmap0, 5)
    print(high, low, cut)

    data = prepare_data(hpxmap,fracdet,fracmin=FRACMIN)
    bkg = fit_bkg_poly(data,sigma=SIGMA)
    #bkg = fit_bkg_gauss(data,sigma=2.0)

    defaults = dict(cmap='gray_r',xsize=args.xsize,smooth=SIGMA)
    for name in PLOT:
        values = DATA[name]
        cls = skymap_factory(PROJ)
        fig = plt.figure(figsize=(16,10))
        smap = cls()

        if COORD == 'gal':
            #print("Transforming to Galactic coordinates...")
            data = data[galpix]
            bkg  = bkg[galpix]


        # Plot just the data
        if name=='data':
            #kwargs.update(vmin=vmin,vmax=vmax,norm=LogNorm())
            #kwargs = dict(defaults,vmin=vmin0,vmax=vmax0)
            kwargs = dict(defaults,vmin=vmin0,vmax=vmax0)
            if args.log: kwargs.update(norm=LogNorm())
            smap.draw_hpxmap(data,**kwargs)

        # Plot just the background
        if name=='bkg':
            kwargs = dict(defaults,vmin=vmin0,vmax=vmax0)
            if args.log: kwargs.update(norm=LogNorm())
            smap.draw_hpxmap(bkg,**kwargs)

        # Plot the data - bkg
        if name=='resid':
            scale = 4*(256./nside)**2
            kwargs = dict(defaults)
            smap.draw_hpxmap((data-bkg),**kwargs)

        # Plot the fractional residual
        if name=='frac':
            scale = 4*(256./nside)**2
            kwargs = dict(defaults,label='SNR/pix',vmin=-scale,vmax=scale)
            smap.draw_hpxmap(((data-bkg)/np.sqrt(bkg)),**kwargs)

        #smap.draw_inset_colorbar(**CBAR_KWARGS.get(PROJ,{}))
        if args.label:
            draw_streams(smap,mod,coord=COORD)
            streamlib.draw_globulars(smap,mod,coord=COORD)

        #plt.ion()
        #import pdb; pdb.set_trace()
        #streamlib.draw_grillmair17(smap,coord=COORD,lw=1)

        plt.suptitle('m-M = %.1f'%mod,y=1.06 if PROJ=='mbpqt' else 1)

        #plt.ion(); plt.show()

        #plt.savefig(os.path.join(movdir,pngfile%mod),bbox_inches='tight')
        pngfile = filebase.format(name,COORD,PROJ)%mod
        pdffile = pngfile.replace('.png','.pdf')%mod
        #print("Saving %s..."%pngfile)
        plt.colorbar()
        plt.savefig(os.path.join(movdir,pngfile))
        plt.savefig(os.path.join(movdir,pdffile))
        #import pdb; pdb.set_trace();
        if not interactive: plt.close('all')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', default = 'iso_hpxcube.fits.gz')
    parser.add_argument('-c','--coord',choices=['gal','cel'],default='cel')
    parser.add_argument('-p','--proj',#choices=['mbptq','splaea','laea'],
                        default='mbptq')
    parser.add_argument('-q','--quad',choices=[1,2,3,4],default=None,type=int)
    parser.add_argument('-d','--data',choices=DATA.keys(),
                        action='append', default=None)
    parser.add_argument('-x','--xsize',default=800,type=int)
    parser.add_argument('-l','--label',action='store_true')
    parser.add_argument('-m','--mod',default=None,type=float)
    parser.add_argument('-s','--step',default=0.3,type=float)
    parser.add_argument('--sigma',default=SIGMA,type=float)
    parser.add_argument('--log',action='store_true')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('-f','--force',action='store_true')
    parser.add_argument('-n','--njobs',default=1,type=int)
    args = parser.parse_args()

    COORD = args.coord
    if args.proj:
        PROJ = args.proj
    elif args.coord == 'cel':
        PROJ = 'mbtpq'
    elif args.coord == 'gal':
        PROJ = 'laea'

    if args.quad is not None: PROJ = 'q%i'%args.quad

    PLOT = args.data if args.data else ['resid']

    FILE = args.filename
    VERSION = FILE.split('_')[-1].replace('.fits.gz','')
    print('version = ', VERSION)
    SIGMA = args.sigma

    print ("Reading %s..."%FILE)
    f = fitsio.FITS(FILE)
    hpxcube = f['HPXCUBE'].read()
    fracdet = np.concatenate(f['FRACDET'].read()['T'])
    modulus = f['MODULUS'].read()
    nside = hp.get_nside(hpxcube[:,0])
    try:
        sfd98 = hp.ud_grade(hp.read_map('data/lambda_sfd_ebv.fits',verbose=False),nside)
    except:
        sfd98 = np.zeros(hp.nside2npix(nside))

    #filebase = '{}-hpxcube_{}_{}_%s.png' #.format(PLOT,COORD,PROJ)
    #movbase  = '{}-hpxcube_{}_{}_{}.gif' #.format(PLOT,COORD,PROJ,VERSION)

    movdir='mov'
    if os.path.exists(movdir):
        subprocess.call('rm -rf %s'%movdir, shell=True)
    mkdir('mov')

    # Distance modulus steps
    if args.mod:
        moduli = [args.mod]
    elif args.step:
        moduli = np.arange(modulus.min(),modulus.max(),args.step)
    else:
        moduli = np.copy(modulus)

    #moduli = [14.5,14.6,14.7,14.8,14.9,15.0,15.1,15.2,15.3,15.4,15.5,15.6,15.7,15.8,15.9,16.0,16.1,16.2,16.3,16.4,16.5]

    # Global vmin,vmax values
    #vmin0,vmax0 = np.percentile(hpxcube[:,0][fracdet > FRACMIN],q=[2,95])
    
    vmin0,vmax0 = np.percentile(hpxcube[:,:][fracdet > FRACMIN],q=[2,95])

    filebase = '{}-hpxcube_{}_{}_%s.png' #.format(PLOT,COORD,PROJ)                                                                                                                                         
    movbase  = 'resid-hpxcube_{}_{}_{}_sigma5.gif'.format(COORD,PROJ,VERSION)

    lon,lat  = hp.pix2ang(nside,np.arange(len(hpxcube[:,0])),lonlat=True)
    galpix   = hp.ang2pix(nside,*gal2cel(lon,lat),lonlat=True)

    #arguments = [(i,mod) for i,mod in enumerate(moduli)]
    arguments = moduli
    if args.njobs < 2:
        map(do_plot,arguments)
    else:
        from multiprocessing import Pool
        pool = Pool(processes=args.njobs,maxtasksperchild=1)
        pool.map(do_plot,arguments)

    print("Lights, Camera, Action...")
    for name in PLOT:
        pngfile = filebase.format(name,COORD,PROJ)%'*'
        movfile = movbase.format(name,COORD,PROJ,VERSION)
        cmd = 'convert -delay 40 -quality 100 %s/%s %s'%(movdir,pngfile,movfile)
        print(cmd)
        subprocess.check_call(cmd,shell=True)

    print("Done.")
