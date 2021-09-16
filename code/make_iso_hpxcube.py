#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os
import glob
import logging
from collections import OrderedDict as odict

import matplotlib as mpl
import __main__ as main
if not hasattr(main, '__file__'): mpl.use('Agg')

import fitsio
import numpy as np
import healpy as hp
import pylab as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LogNorm
import scipy.ndimage as nd
from scipy.interpolate import interp1d

from utils import load_infiles
from ugali.utils import healpix
from ugali.isochrone import factory as isochrone_factory
#from ugali.utils.binning import take2D
from ugali.utils.logger import logger
from ugali.utils.projector import angsep

from streamlib import make_mask
from isolib import *
import pal13files
import filter_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f','--filename',default='test_hpxcube_pal13.fits.gz')
    parser.add_argument('-a','--age',default=AGE,type=float)
    parser.add_argument('-z','--metallicity',default=Z,type=float)
    parser.add_argument('-n','--nside',default=NSIDE,type=int)
    parser.add_argument('-c','--color',nargs=2,default=[CMIN,CMAX],type=float)
    parser.add_argument('-m','--mag',nargs=2,default=(MMIN,MMAX),type=float)
    parser.add_argument('-u','--mod',nargs=2,default=(MODMIN,MODMAX),type=float)
    parser.add_argument('-s','--step',default=MSTEP,type=float)
    parser.add_argument('--color-shift',nargs=2,default=COLOR_SHIFT,type=float)
    parser.add_argument('--error-factor',default=ERROR_FACTOR,type=float)
    parser.add_argument('--mash',default=1,type=int)
    parser.add_argument('--nbins',default=151,type=int)
    parser.add_argument('--imf',action='store_true')
    parser.add_argument('--rgb-clip',default=None,type=float)
    parser.add_argument('--src',default='cookie',type=lambda s : s.lower())
    parser.add_argument('--sigma',default=5,type=int)
    parser.add_argument('-v','--verbose',action='store_true')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(level)
    nside = args.nside

    AGE          = args.age
    Z            = args.metallicity
    MMIN,MMAX    = args.mag
    CMIN,CMAX    = args.color
    MASH         = args.mash
    NBINS        = args.nbins
    IMF          = args.imf
    RGB_CLIP     = args.rgb_clip
    COLOR_SHIFT  = args.color_shift
    ERROR_FACTOR = args.error_factor
    MSTEP        = args.step
    MODMIN,MODMAX= args.mod
    MODULUS      = np.arange(MODMIN,MODMAX,MSTEP)
    if MODMAX not in MODULUS: MODULUS = np.append(MODULUS,MODMAX)

    CBINS = np.linspace(CMIN,CMAX,NBINS)
    MBINS = np.linspace(MMIN,MMAX,NBINS)

    logger.info("Reading coverage fraction...")
    dirname = 'decals/'
    fracdet = fitsio.read(dirname+'decals_dr7_pseudo_fracdet.fits.gz')
    scale = (4096/nside)**2
    #pix = hp.nest2ring(nside,frac['PIXEL']//scale)

    #fracdet = np.zeros(hp.nside2npix(nside))
    #np.add.at(fracdet,pix,frac['SIGNAL'])
    #fracdet /= scale

    logger.info("Reading catatogs...")
    DIRNAME = '/data/des81.b/data/tavangar/south_skim/'
    #filenames = sorted(glob.glob(DIRNAME+'*.fits'))
    filenames = pal13files.get_files(346.48, 12.76)
    print(filenames)
    data = load_infiles(filenames,columns=COLUMNS,multiproc=8) # nora has 16
    logger.info("Catalogs loaded...")

    # Select magnitude range
    #if MASH != 2:
    #    print "Selecting: EXTENDED_CLASS <= %i"%(MASH)
    #    data = data[data[EXT] <= MASH]
    print("Selecting: %.1f < %s < %.1f"%(MMIN,MAG_G,MMAX))
    data = data[(data[MAG_G] < MMAX) & (data[MAG_G] > MMIN)]
    print("Selecting: %.1f < %s - %s < %.1f"%(CMIN,MAG_G,MAG_R,CMAX))
    data = data[(data[MAG_G]-data[MAG_R] < CMAX)&(data[MAG_G]-data[MAG_R] > CMIN)]

    if args.src not in ['cookie']:
        print("Creating background CMD...")
        sample = int(10)
        d = data[::sample]
        _,_,bkg,bkg_smooth = make_bkg_cmd(d[MAG_R],d[MAG_G],d['RA'],d['DEC'],
                                          sigma=2)
        _,_,sig,sig_smooth = make_sig_cmd(name=args.src,sigma=args.sigma)

        fig,ax = plt.subplots(1,3,figsize=(12,5))
        plt.sca(ax[0])
        plt.pcolormesh(CBINS,MBINS,bkg_smooth,norm=LogNorm(),
                       vmin=1e-3*bkg_smooth.max()); 
        plt.colorbar()
        plt.xlabel('g-r'); plt.ylabel('g')
        ax[0].invert_yaxis()

        plt.sca(ax[1])
        plt.pcolormesh(CBINS,MBINS,sig_smooth,norm=LogNorm(),
                       vmin=1e-3*sig_smooth.max()); 
        plt.colorbar()
        plt.xlabel('g-r'); plt.ylabel('g')
        ax[1].invert_yaxis()

        #snr2 = sig_smooth**2/bkg_smooth
        snr2 = sig_smooth/bkg_smooth
        snr2 = np.ma.array(snr2,mask=(~np.isfinite(snr2))|(snr2==0))
        plt.sca(ax[2])
        plt.pcolormesh(CBINS,MBINS,snr2,norm=LogNorm(),vmin=1e-3*snr2.max())
        plt.colorbar()
        plt.xlabel('g-r'); plt.ylabel('g')
        ax[2].invert_yaxis()

        plt.suptitle(args.src.upper())
        plt.savefig('%s_matched_filter.png'%args.src,bbox_inches='tight')

    hpxcube = np.zeros((hp.nside2npix(nside),len(MODULUS)))

    for i,mu in enumerate(MODULUS):
        #if mu > 17: break
        print(" bin=%i: m-M = %.1f..."%(i,mu))

        #if args.src in ['cookie']:
        #    _,_,cookie,path = cookie_cutter(mu=mu, age=AGE, z=Z, dmu=DMU, 
        #                                    color_shift=COLOR_SHIFT, 
        #                                    error_factor=ERROR_FACTOR,
        #                                    imf=IMF,rgb_clip=RGB_CLIP)
        #    matched = cookie
        #else:
        #    _,_,sig,sig_smooth = make_sig_cmd(name=args.src,mod=mu,sigma=args.sigma)
        #    #matched = sig_smooth**2/bkg_smooth
        #    matched = sig_smooth/bkg_smooth
        #    matched[~np.isfinite(matched)] = 0

        #save_matched_filter('plots/%s_matched_m%.1f.png'%(args.src,mu),
        #                    '%s (%.1f)'%(args.src.upper(),mu),matched)

        #weight = take2d(matched,data[MAG_G]-data[MAG_R],data[MAG_G],CBINS,MBINS)
        #weight = np.nan_to_num(weight)

        #sel = weight > 0 

        gmin = 20.2 - (16.8 - mu)

        sel1 = filter_data.select_isochrone(data[MAG_G], data[MAG_R], err=None, iso_params=[
            mu, AGE, Z], C=COLOR_SHIFT, E=ERROR_FACTOR, gmin=gmin, survey='DECaLS')
        sel2 = filter_data.select_isochrone_grz(data[MAG_G], data[MAG_R], data[MAG_Z], err=None, iso_params=[
            mu, AGE, Z], C=COLOR_SHIFT, E=ERROR_FACTOR, gmin=gmin, survey='DECaLS')
        sel = sel1 & sel2
        d = data[sel]
        #w = weight[sel]
        pixel = healpix.ang2pix(nside,d['RA'],d['DEC'])
        pix, cts = np.unique(pixel, return_counts=True)
        #np.add.at(cts,pixel,w)
        hpxcube[pix,i] = cts


    print("Writing %s..."%args.filename)
    header = healpix.header_odict(nside,coord='C',partial=False).values()

    header += [
        odict(name='CSHIFT',value=str(COLOR_SHIFT),
              comment='Isochrone selection color shift'),
        odict(name='EFACTOR',value=ERROR_FACTOR,
              comment='Isochrone selection error factor'),
        odict(name='DMU',value=DMU,
              comment='Isochrone selection delta distance modulus'),
        odict(name='AGE',value=AGE,
              comment='Isochrone age'),
        odict(name='Z',value=Z,
              comment='Isochrone metallicity'),
        ]
    f = fitsio.FITS(args.filename,'rw',clobber=True)
    print("  Writing hpxcube...")
    f.write(hpxcube,header=header,extname='HPXCUBE')
    print("  Writing fracdet...")
    f.write(fracdet,extname='FRACDET',header=header)
    print("  Writing bins...")
    f.write(MODULUS,extname='MODULUS',header={'DMU':DMU})
    f.close()
