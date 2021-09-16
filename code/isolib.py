#!/usr/bin/env python
"""
Tools for working with the data
"""
__author__ = "Alex Drlica-Wagner"

import os
import glob
import logging
from collections import OrderedDict as odict

import matplotlib as mpl
#import __main__ as main
#if not hasattr(main, '__file__'): mpl.use('Agg')

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
from ugali.utils.shell import mkdir

#from streamlib import make_mask

#DIRNAME = '/data/des81.b/data/tavangar/skim'
DIRNAME = '/data/des40.b/data/decals/dr8/south_skim'

#MAG_G = 'SOF_PSF_MAG_CORRECTED_G'
#MAG_R = 'SOF_PSF_MAG_CORRECTED_R'
#EXT = 'EXT_SOF'
MAG_G = 'MAG_SFD_G'
MAG_R = 'MAG_SFD_R'
MAG_Z = 'MAG_SFD_Z'
EXT = 'EXTENDED_CLASS'
COLUMNS = ['RA','DEC',MAG_G,MAG_R,MAG_Z,EXT]

NSIDE = 512
MASH = 1
MMIN,MMAX = 20.2,23.
CMIN,CMAX = 0.0,1.0
MODMIN,MODMAX = 16,17.6
MSTEP = 0.1
NBINS = 151
CBINS = np.linspace(CMIN,CMAX,NBINS)
MBINS = np.linspace(MMIN,MMAX,NBINS)

# Isochrone parameters
AGE = 11.0
Z = 0.001
DMU = 0.5
COLOR_SHIFT=[0.01,0.05]
ERROR_FACTOR=[3.0]
# Absolute magnitude of MSTO 
# WARNING: This is hardcoded
MSTO = 3.5

def mkpol(mu, age=AGE, z=Z, dmu=DMU, C=COLOR_SHIFT, E=ERROR_FACTOR, err_type='median', rgb_clip=None):
    """ Builds ordered polygon for masking """
    A1 = 0.0010908679647672335
    A2 = 27.091072029215375
    A3 = 1.0904624484538419
    err = lambda x: A1 + np.exp((x - A2) / A3) # median
    iso = isochrone_factory('Dotter2008', age=age, z=z)
    c = iso.color
    m = iso.mag
    if rgb_clip is not None:
        rgb_mag = (MSTO-rgb_clip)
        print ("Clipping RGB at mag = %g"%rgb_mag)
        c = c[m > rgb_mag]
        m = m[m > rgb_mag]

    mnear = m + mu - dmu / 2.
    mfar = m + mu + dmu / 2.

    C0,C1 = C
    COL = np.r_[c + E * err(mfar) + C1, c[::-1] - E * err(mnear[::-1]) - C0]
    MAG = np.r_[m, m[::-1]] + mu
    return np.c_[COL, MAG]

def iso_sel(g, r, mu, age, z, dmu=DMU, color_shift=COLOR_SHIFT, error_factor=ERROR_FACTOR):
    """ Perform isochrone selection. """
    mk = mkpol(mu=mu, age=age, z=z, dmu=0.5, C=color_shift, E=error_factor)
    pth = Path(mk)
    cm = np.vstack([g - r, g - mu]).T
    return pth.contains_points(cm)


def cookie_cutter(mu=16.0, age=AGE, z=Z, dmu=DMU, 
                  color_shift=COLOR_SHIFT, error_factor=ERROR_FACTOR,
                  imf=False,rgb_clip=None):
                 
    """ Create a cookie cutter...

    Parameters:
    -----------
    mu  : distance modulus (m-M)
    age : isochrone age (Gyr)
    z   : isochrone metallicity
    dmu : distance modulus spread
    color_shift : [left,right] broadening of selection in color-space
    error_factor : number of 'sigma' to broaden selection by

    Returns:
    --------
    cbins,mbins,weight,path : color bins, mag bins, weight hist, selection path
    """
    CCENT = (CBINS[1:]+CBINS[:-1])/2.
    MCENT = (MBINS[1:]+MBINS[:-1])/2.
    cc,mm = np.meshgrid(CCENT,MCENT)

    vertices = mkpol(mu=mu, age=age, z=z, dmu=0.5, C=color_shift, E=error_factor, rgb_clip=rgb_clip)
    path = Path(vertices)
    sel = path.contains_points(np.vstack([cc.flat,mm.flat]).T).reshape(cc.shape)

    weight = np.zeros_like(cc)
    weight[np.where(sel.reshape(cc.shape))] = 1.0

    # Uncomment to implement the imf weighting
    if imf:
        iso = isochrone_factory('Dotter2008', age=age, z=z, distance_modulus=mu)
        mass_init,mass_pdf,mass_act,mag_1,mag_2 = iso.sample()
        pdf = interp1d(mag_1+mu, mass_pdf, bounds_error=False, fill_value=np.nan)
        weight *= np.nan_to_num(pdf(mm))
        # This is a bit dicey... normalizing to observed stars
        weight /= (weight.sum() * CBINS.ptp() * MBINS.ptp())

    return CBINS,MBINS,weight,path

def cookie_weight(g, r, mu=16.0, age=AGE, z=Z, dmu=DMU, 
                  color_shift=COLOR_SHIFT, error_factor=ERROR_FACTOR):
    cbins,mbins,weight = cookie(mu,age,z,dmu,color_shift,error_factor)
    w = take2D(weight,g-r,g,cbins,mbins)
    return np.nan_to_num(w)

def plot_cookie(cbins,mbins,cookie,path=None):
    plt.figure(figsize=(5,6))
    vals = np.ma.array(cookie,mask=cookie<=0)
    plt.pcolormesh(cbins,mbins,vals,norm=LogNorm())
    plt.colorbar(label='Weight')
    if path:
        patch = PathPatch(path,edgecolor='b',facecolor='none')
        plt.gca().add_artist(patch)
    plt.ylim(MMAX,MMIN); plt.xlim(CMIN,CMAX)
    plt.xlabel(r'$(g-r)$',fontsize=14); plt.ylabel(r'$g$',fontsize=14)
    #plt.title('m-M = %.1f'%mu)
    #plt.savefig('cookie_cutter_m%.1f.png'%mu,bbox_inches='tight')
    #plt.close()
    
def make_bkg_cmd(g, r, ra, dec, sigma=5):
    pix = hp.ang2pix(nside,ra,dec,lonlat=True)
    sel = ~make_mask(nside)[pix]
    g = g[sel]
    r = r[sel]
    num,_,_ = np.histogram2d(g-r,g,bins=[CBINS,MBINS])
    smooth = nd.gaussian_filter(num,sigma=sigma)
    num /= num.sum()
    smooth /= smooth.sum()
    return CBINS,MBINS,num.T,smooth.T

def make_sig_cmd(name='ngc7089',mod=None, sigma=5):
    """ Make the signal CMD (usually for globular clusters).

    Parameters:
    -----------
    name  : name of the object to build from.
    mod   : distance modulus
    sigma : cmd Gauassian smoothing radius

    Returns:
    --------
    color_bis,mag_bins,cts,smooth_cts
    """
    if name in ['ngc0288','ngc288']:
        return make_ngc0288_cmd(mod,sigma)
    elif name in ['ngc1261']:
        return make_ngc1261_cmd(mod,sigma)
    elif name in ['ngc1851']:
        return make_ngc1851_cmd(mod,sigma)
    elif name in ['ngc1904']:
        return make_ngc1904_cmd(mod,sigma)
    elif name in ['ngc7089','m2']:
        return make_ngc7089_cmd(mod,sigma)
    else:
        msg = "Unrecognized name: %s"%name
        raise ValueError(msg)

def load_data(ra,dec,radius=0.25):
    pix = healpix.ang2disc(32, ra, dec, radius=radius, inclusive=True)
    filenames = []
    for p in pix:
        filenames += glob.glob(DIRNAME+'/y3a2_ngmix_cm_%05d.fits'%p)
    print("Loading %i files..."%len(filenames))
    data = load_infiles(filenames,columns=COLUMNS)
    sep = angsep(ra, dec, data['RA'],data['DEC'])
    return data[sep < radius]

load_gc_data = load_data

def make_cmd(ra,dec,delta_mod=0,sigma=5):
    """ 
    Create a signal CMD from a globular cluster.

    Parameters:
    -----------
    ra  : ra of gc
    dec : dec of gc
    mod : distance modulus
    sigma : smoothing
    """
    data = load_gc_data(ra,dec,radius=0.25)
    sep = angsep(ra, dec, data['RA'],data['DEC'])
    d = data[(sep > 0.07) & (sep < 0.12)]
    g = d[MAG_G] + delta_mod
    r = d[MAG_R] + delta_mod
    num,_,_ = np.histogram2d(g-r,g,bins=[CBINS,MBINS])
    smooth = nd.gaussian_filter(num,sigma=sigma)
    num /= num.sum()
    smooth /= smooth.sum()
    return CBINS,MBINS,num.T,smooth.T
make_gc_cmd = make_cmd

def make_ngc0288_cmd(mod=None, sigma=5):
    """ Make signal CMD for NGC 288 """
    RA, DEC, MOD = 13.197706, -26.589901, 14.95
    if mod is None: mod = MOD
    return make_gc_cmd(RA,DEC,mod - MOD, sigma)

def make_ngc1261_cmd(mod=None, sigma=5):
    """ Make signal CMD for NGC 1261 """
    RA, DEC, MOD = 48.063930, -55.216809, 16.11
    if mod is None: mod = MOD
    return make_gc_cmd(RA,DEC,mod - MOD, sigma)

def make_ngc1851_cmd(mod=None, sigma=5):
    """ Make signal CMD for NGC 1851 """
    RA, DEC, MOD = 78.528042, -40.046611, 15.53
    if mod is None: mod = MOD
    return make_gc_cmd(RA,DEC,mod - MOD, sigma)

def make_ngc1904_cmd(mod=None, sigma=5):
    """ Make signal CMD for NGC 1904 """
    RA, DEC, MOD = 81.0462, -24.52472, 15.56
    if mod is None: mod = MOD
    return make_gc_cmd(RA,DEC,mod - MOD, sigma)

def make_ngc7089_cmd(mod=None, sigma=5):
    """ Make signal CMD for NGC 7089 (M2) """
    RA, DEC, MOD = 323.362552, -0.823318, 15.46
    if mod is None: mod = MOD
    return make_gc_cmd(RA,DEC,mod - MOD, sigma)

make_m2_cmd = make_ngc7089_cmd

def plot_matched_filter(matched,title):
    plt.figure()
    matched = np.ma.array(matched,mask=(~np.isfinite(matched))|(matched==0))

    plt.pcolormesh(CBINS,MBINS,matched,norm=LogNorm(),
                   vmax = 20., vmin = 20. * 1e-2)
    plt.colorbar()
    plt.xlabel('g-r'); plt.ylabel('g')
    plt.gca().invert_yaxis()

    if title: plt.suptitle(title)

def save_matched_filter(filename,title,matched):
    outdir = os.path.dirname(filename)
    mkdir(outdir)
    plot_matched_filter(matched,title=title)
    plt.savefig(filename,bbox_inches='tight')
    plt.close()
    
def take2d(a,x,y,xbins,ybins):
    """ 
    Take values from a histgram based on the x,y values.  Returns
    np.nan when x,y values are outside of the range the histogram is
    defined on.
    
    Parameters:
    -----------
    a     : histogram to take from; a.shape = (n,m)
    x     : x-values to take
    y     : y-values to take
    xbins : bin edges in x-dimension len(xbins) = n+1
    ybins : bin edges in y-dimension len(ybins) = m+1

    Returns:
    --------
    values : values taken from histogram
    """
    # Add overflow and underflow bins
    hist = np.nan*np.ones((a.shape[0]+2,a.shape[1]+2))
    hist[1:-1,1:-1] = a

    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    xbins[-1] += 1.e-10 * (xbins[-1] - xbins[-2])
    ybins[-1] += 1.e-10 * (ybins[-1] - ybins[-2])

    xidx = np.digitize(x,xbins)
    yidx = np.digitize(y,ybins)

    return hist[yidx,xidx]

def load_stream_data(name):
    import results
    import streamlib
    stream = streamlib.load_streams()[name]
    stream = results.create_result(stream)
    #rough endpoint
    ra0,dec0 = np.array(stream['ends']).mean(axis=0)
    length = stream['length_deg']
    return load_data(ra0,dec0,length)

def select_stream_data(name,data=None, **kwargs):
    import streamlib
    stream = streamlib.load_streams()[name]

    kwargs['mu'] = stream['modulus']
    kwargs['age'] = stream['age']
    kwargs['z'] = stream['metallicity']
    cbins,mbins,cookie,path = cookie_cutter(**kwargs)

    if data is None:
        data = load_stream_data(name)

    print("Performing isochrone selection...")
    weight = take2d(cookie,data[MAG_G]-data[MAG_R],data[MAG_G],CBINS,MBINS)
    weight = np.nan_to_num(weight)
    sel = weight > 0 
    return data[sel]
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
