#!/usr/bin/env python                                                                                                                                                                                       
"""                                                                                                                                                                                                         
This is a sandbox for plotting field of streams isochrone selections.                                                                                                                                       
"""
__author__ = "Kiyan Tavangar"

import os, sys
import subprocess
from collections import OrderedDict as odict

import matplotlib as mpl
import __main__ as main
interactive = hasattr(main, '__file__')
if not interactive: mpl.use('Agg')

import fitsio as fits
import healpy as hp
import numpy as np
import pylab as plt
from matplotlib.colors import LogNorm
sys.path.append('/data/des81.b/data/tavangar/streams/code')
import density_plot

def plot():
    #i = np.abs(modulus-mod).argmin()
    mod = [14.0, 14.3, 14.6, 14.9, 15.2, 15.5, 15.8, 16.1, 16.4, 16.7, 17.0, 17.3, 17.6, 17.9, 18.2, 18.5, 18.8, 19.1, 19.4, 19.7, 20.0]
    for i in range(0,20):
        modu = mod[i]
        print("%i : m-M = %.1f"%(i,modu))

        hpxmap = np.copy(hpxcube[:,i])
    #data,bkg = fit_data_bkg(hpxmap,fracdet,fracmin=FRACMIN,sigma=SIGMA)                                                                                                                                    
        data = density_plot.prepare_hpxmap(hpxmap)
        bkg=0

        #hpxmap[hpxmap == 0] = np.nan
        #cut = np.nanpercentile(hpxmap, 99)
        #hpxmap[hpxmap >cut] = np.nan
        #means = np.nanmean(hpxmap)
        #print(means)
        #high = np.nanpercentile(hpxmap, 95)
        #low = np.nanpercentile(hpxmap, 5)
        #print(high, low, cut)
    
        smap = plot_density(data, bkg, center=(lon, lat), filename='decals/density_maps/pal13_density_{}.png'.format(int(10*mu)))
        #hp.mollview(hpxmap+1, cmap='gray_r', min=low, max=high, norm='log')
    
        plt.suptitle('m-M = %.1f'%modu)
        pngfile = filebase.format()%modu
        plt.savefig(os.path.join(movdir,pngfile))
        #plt.savefig(os.path.join(movdir,pdffile)
        #plt.show()

sigma=0.1
vmin0, vmax0 = 0,20
lon, lat = 346, 12
if __name__ == "__main__":
    movdir = 'mov'                                                                                                                                         
    movbase  = 'resid-hpxcube_mov.gif'
    hpxcube, fracdet, modulus = density_plot.load_hpxcube('decals/iso_hpxcube_pal13.fits.gz')
    mod = [14.0, 14.3, 14.6, 14.9, 15.2, 15.5, 15.8, 16.1, 16.4, 16.7, 17.0, 17.3, 17.6, 17.9, 18.2, 18.5, 18.8, 19.1, 19.4, 19.7, 20.0]
    for i in range(0,20):
        mu = mod[i]
        print('modulus = {}'.format(mu))                                                                                                                                                                 
        data = density_plot.prepare_hpxmap(mu, hpxcube, fracdet, modulus)
        bkg = 0
        #bkg = fit_background(data, center=(lon,lat), coords='cel', sigma=sigma, deg=5)                                                                                                                  
        smap = density_plot.plot_density(data, bkg, center=(lon, lat), filename='decals/density_maps/pal13_density_{}.png'.format(int(10*mu)))
        plt.suptitle('m-M = %.1f'%mu)
        pngfile = filebase.format()%mu
        plt.savefig(os.path.join(movdir,pngfile))

    pngfile = filebase.format()%'*'
    movfile = movbase
    cmd = 'convert -delay 40 -quality 100 %s/%s %s'%(movdir,pngfile,movfile)
    print(cmd)
    subprocess.check_call(cmd,shell=True)
