import numpy as np
import healpy as hp
import fitsio as fits
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import skymap
from skymap.utils import cel2gal, gal2cel
from skymap.utils import setdefaults
from streamlib import skymap_factory
import streamlib

#def plot_pretty(dpi=175, fontsize=15, labelsize=15, figsize=(3, 3)):
    # import pyplot and set some parameters to make plots prettier

#    plt.rc('savefig', dpi=dpi)
    #plt.rc('text', usetex=True)
    #plt.rc('font', size=fontsize)
    #plt.rc('xtick.major', pad=5)
    #plt.rc('xtick.minor', pad=5)
    #plt.rc('ytick.major', pad=5)
    #plt.rc('ytick.minor', pad=5)
    #plt.rc('figure', figsize=figsize)

    #mpl.rcParams['xtick.labelsize'] = labelsize
    #mpl.rcParams['ytick.labelsize'] = labelsize
#    mpl.rcParams.update({'figure.autolayout': True})

def load_hpxcube(filename):
    print("Reading %s..." % filename)
    f = fits.FITS(filename)
    hpxcube = f['HPXCUBE'].read()
    #try:
        #fracdet = f['FRACDET'].read()
        #print('fracdet test', np.sum(fracdet > 0.5))
    #except:
    print('Skipping fracdet...')
    fracdet = np.zeros_like(hpxcube[:, 0])
    fracdet[np.where(hpxcube[:, 0] > 0)] = 1
    try:
        modulus = f['MODULUS'].read()
    except:
        print('Error reading modulus...')
        modulus = np.array([16.])

    return hpxcube, fracdet, modulus

def prepare_hpxmap(mu, hpxcube, fracdet, modulus, fracmin=0, clip=100, sigma=0.2, **mask_kw):
    i = np.argmin(np.abs(mu - modulus))
    hpxmap = np.copy(hpxcube[:, i])
    data = hpxmap
    #data = streamlib.prepare_data(hpxmap, fracdet, fracmin=fracmin, clip=clip, mask_kw=mask_kw)
    # bkg = streamlib.fit_bkg_poly(data, sigma=sigma)
    # bkg = None
    return data

def plot_density(data, bkg, coords='cel', coord_stream=None, center=(0,0), proj='q2', filename=None, smap=None, 
                 vmin = 0, vmax = 10, **kwargs):
    defaults = dict(cmap='gray_r', xsize=2000, smooth=0.1)
    setdefaults(kwargs, defaults)

    nside = hp.get_nside(data)

    if smap is None:
        plt.figure()
        #smap = skymap.Skymap(projection=proj, lon_0=center[0], lat_0=center[1], celestial=False)
        smap = skymap_factory(proj)
        smap = smap()

    if coords == 'gal':
        lon, lat = hp.pix2ang(nside, np.arange(len(data)), lonlat=True)
        galpix = hp.ang2pix(nside, *gal2cel(lon, lat), lonlat=True)
        # smap.draw_hpxmap((data - bkg)[galpix], **kwargs)
        # IS THIS RIGHT?
        smap.draw_hpxmap((data[galpix] - bkg), **kwargs)
        # smap.draw_hpxmap(data[galpix], **kwargs)
    elif coords == 'stream':
        if not coord_stream:
            print('Need to input coord_stream!')
        streampix = streamlib.get_streampix(data=data, stream=coord_stream)
        smap.draw_hpxmap((data[streampix] - bkg), **kwargs)

    else:
        smap = smap.draw_hpxmap((data - bkg), **kwargs)
        # smap.draw_hpxmap(data, **kwargs)

    if filename:
        plt.savefig(filename)
    
    smap
    return smap

def fit_background(data, center=(0,0), coords='cel', coord_stream=None, proj='ortho', sigma=0.2, percent=[2, 95], deg=5):
    bkg = streamlib.fit_bkg_poly(data, center=center, coords=coords,
                                 coord_stream=coord_stream, proj=proj, sigma=sigma, percent=percent, deg=deg)
    return bkg

#vmin0, vmax0 = 0, 15

if __name__ == '__main__':
    #plot_pretty()
    hpxcube, fracdet, modulus = load_hpxcube('decals/iso_hpxcube_pal13.fits.gz')
    print('modulus = {}'.format(modulus))
    mu = 16.8
    sigma=0.1
    vmin0, vmax0 = 0, 20
    #lon, lat = 20, -30
    lon, lat = 346, 12
    data = prepare_hpxmap(mu, hpxcube, fracdet, modulus)
    bkg = 0
    #bkg = fit_background(data, center=(lon,lat), coords='cel', sigma=sigma, deg=5)
    smap = plot_density(data, bkg, center=(lon, lat), filename='decals/density_maps/pal13_density_{}.png'.format(int(10*mu)))
