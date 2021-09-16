#!/usr/bin/env python
"""
Generic python script.
"""
__authors__ = "Alex Drlica-Wagner, Kiyan Tavangar"

import sys
import glob

import numpy as np
from PIL import Image, ImageDraw
import healpy as hp
import fitsio as fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.colors as colors
import scipy.ndimage as nd
import fitsio

import skymap, skymap.survey, skymap.core
from skymap.core import Skymap
from skymap.utils import cel2gal, gal2cel
from skymap.utils import setdefaults

from streamlib import skymap_factory
import streamlib

from polyfit2d import curvefit2d
from polyfit2d import polyfit2d
from numpy.polynomial import polynomial

RA,DEC = 346.685333, 12.770972
RA -= 360*(RA>180)
RA_top,DEC_top = -9.8,18.2
RA_bottom,DEC_bottom = -15.7,8.9

ras = np.linspace(RA_bottom, RA_top, 50)
decs = 28.91 + 0.97*ras - 0.02*(ras**2)

pmra, pmdec = 0.80, 1.45

# plot ridgeline
smap=skymap_factory('pal13')
lon,lat = smap().draw_great_circle(RA_bottom, DEC_bottom, RA_top, DEC_top, 'short')
lon1,lat1 = smap().draw_great_circle(RA_bottom, DEC_bottom, RA, DEC, 'full')
lon2,lat2 = smap().draw_great_circle(RA, DEC, RA_top, DEC_top, 'short')

# plot proper motion    
x1, y1 = RA, DEC
x2, y2 = RA+pmra, DEC+pmdec

# coords of northern spur
RA_north_bottom, DEC_north_bottom = -6.2, 19.9
RA_north_top, DEC_north_top = -4.0, 26
lon_spur,lat_spur = smap().draw_great_circle(RA_north_bottom, DEC_north_bottom, RA_north_top, DEC_north_top, 'short')

def plot_pretty(dpi=175, fontsize=15, labelsize=15, figsize=(10, 8), tex=True):
    # import pyplot and set some parameters to make plots prettier
    plt.rc('savefig', dpi=dpi)
    plt.rc('text', usetex=tex)
    plt.rc('font', size=fontsize)
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('figure', figsize=figsize)
    mpl.rcParams['xtick.labelsize'] = labelsize
    mpl.rcParams['ytick.labelsize'] = labelsize
    mpl.rcParams.update({'figure.autolayout': True})
    
plot_pretty(fontsize=15)

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
    #mask = np.load('pal13/Pal13_mask.npy')
    #m = np.where(mask == 1)
    #hpxmap[m] = 2.810558765123279
    #data = streamlib.prepare_data(hpxmap, fracdet, fracmin=fracmin, clip=clip, mask_kw=mask_kw)
    # bkg = streamlib.fit_bkg_poly(data, sigma=sigma)
    # bkg = None
    return data

def fit_bkg_poly(data, lon,lat,sigma=0.1, percent=[2, 95], deg=5):
    """ Fit foreground/background with a polynomial """
    #nside = hp.get_nside(data)
    #lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    #lon -= 360*(lon > 180)
    vmin, vmax = np.percentile(data, q=percent)
    data = np.clip(data, vmin, vmax)

    #data.fill_value = np.ma.median(data)
    #data = np.ma.array(hp.smoothing(data, sigma=np.radians(sigma), verbose=False))
    smap = skymap.survey.DECaLSSkymapPal13(parallels=False, meridians=False)

    #sel = ~data.mask
    #x, y = smap(lon, lat)
    #v = data
    #c = polyfit2d(x, y, v, [deg, deg])

    # Select just the active area
    sel = (  (lon > smap.FRAME[0][1]) & (lon < smap.FRAME[0][0])) \
          & ((lat > smap.FRAME[1][0]) & (lat < smap.FRAME[1][1]))
    x, y = smap(lon[sel], lat[sel])
    v = data[sel]
    coeff = polyfit2d(x, y, v, [deg, deg])

    # Evaluate the polynomial
    x, y = smap(lon, lat)
    bkg = polynomial.polyval2d(x, y, coeff)
    #bkg = np.ma.array(bkg, fill_value=np.nan)
    return bkg

def draw_image(xx,yy,data,**kwargs):
    ax = plt.gca()
    kwargs.update(vmin=-0.8,vmax=0.6, cmap='gray_r')
    ax.pcolormesh(xx,yy,data,**kwargs)
    ax.set_aspect('equal')
    ax.set_title(r'$\mathrm{Spatial\ Plot}$')
    ax.invert_xaxis()
    ax.set_xlim(5,-35)
    ax.set_ylim(-5,33)
    ax.set_ylabel(r'$\mathrm{Dec\ (deg)}$')
    ax.set_xlabel(r'$\mathrm{RA\ (deg)}$')
    #plt.grid('on',ls=':',color='gray')
    
def draw_all(xx,yy,data,dust,bkg,**kwargs):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    
    kwargs.update(vmin=-0.8,vmax=0.5)
    ax1.pcolormesh(xx,yy,data-bkg,**kwargs)
    ax1.scatter(RA,DEC,c='r',s=15)
    xline = [RA_bottom, RA_top]
    yline = [DEC_bottom, DEC_top]
    #ax1.plot(xline,yline, c='b', linewidth=0.5)
    #ax1.plot(ras, decs, c='r', linewidth=0.5)
    ax1.set_aspect('equal')
    #ax1.plot(lon,lat, c='r', linewidth=1.5, linestyle='dashed')
    #ax1.plot(lon1,lat1, c='b', linewidth=1.5, linestyle='dashed')
    ax1.set_title(r'$\mathrm{Spatial\ Plot}$')
    ax1.invert_xaxis()
    ax1.set_xlim(5,-35)
    ax1.set_ylim(-5,33)
    ax1.set_ylabel(r'$\mathrm{Dec\ (deg)}$')
    ax1.set_xlabel(r'$\mathrm{RA\ (deg)}$')
    #plt.grid('on',ls=':',color='gray')
    
    kwargs.update(vmin=0,vmax=9)
    ax2.pcolormesh(xx,yy,bkg,**kwargs)
    ax2.scatter(RA,DEC,c='r',s=15)
    #ax2.scatter(RA_top,DEC_top,c='b',s=15)
    #ax2.scatter(RA_bottom,DEC_bottom,c='b',s=15)
    #ax2.plot(xline,yline, c='r', linewidth=1.5)
    ax2.set_aspect('equal')
    ax2.set_title(r'$\mathrm{Background}$')
    ax2.plot(lon,lat, c='r', linewidth=1.5, linestyle='dashed')
    ax2.plot(lon_spur,lat_spur, c='r', linewidth=1.5, linestyle='dotted')
    #ax2.plot(lon1,lat1, c='b', linewidth=1.5, linestyle='dashed')
    #ax2.plot(lon2,lat2, c='b', linewidth=1.5, linestyle='dashed')
    ax2.arrow(x1, y1, x2-x1, y2-y1, head_width=1, color='r')
    ax2.invert_xaxis()
    ax2.set_xlim(5,-35)
    ax2.set_ylim(-5,33)
    ax2.set_ylabel(r'$\mathrm{Dec\ (deg)}$')
    ax2.set_xlabel(r'$\mathrm{RA\ (deg)}$')
    #plt.grid('on',ls=':',color='gray')
    
    kwargs.update(vmin=0,vmax=0.2)
    ax3.pcolormesh(xx,yy,dust,**kwargs)
    ax3.scatter(RA,DEC,c='r',s=15)
    #ax3.scatter(RA_top,DEC_top,c='b',s=15)
    #ax3.scatter(RA_bottom,DEC_bottom,c='b',s=15)
    #ax3.plot(xline,yline, c='r', linewidth=1.5)
    ax3.set_aspect('equal')
    ax3.set_title(r'$\mathrm{SFD\  Dust\  Map}$')
    ax3.invert_xaxis()
    ax3.plot(lon,lat, c='r', linewidth=1.5, linestyle='dashed')
    ax3.plot(lon_spur,lat_spur, c='r', linewidth=1.5, linestyle='dotted')
    ax3.arrow(x1, y1, x2-x1, y2-y1, head_width=1, color='r')
    ax3.set_xlim(5,-35)
    ax3.set_ylim(-5,33)
    ax3.set_ylabel(r'$\mathrm{Dec\ (deg)}$')
    ax3.set_xlabel(r'$\mathrm{RA\ (deg)}$')
    #plt.grid('on',ls=':',color='gray')


def make_map_vec(theta, phi, data):
    assert len(theta) == len(phi) == len(data)
    e1map = np.full(hp.nside2npix(512), 0, dtype=np.float)
    index = hp.ang2pix(512, theta, phi, lonlat=True)
    values = np.fromiter((np.mean(data[index==i]) for i in np.unique(index)), float, count=len(np.unique(index)))
    e1map[np.unique(index)] = values
    return e1map

def run(hpxmap, dust=None, planck=None, sigma=0.3,**kwargs):
    defaults = dict(cmap='gray_r')
    setdefaults(kwargs, defaults)

    pixscale = 0.1 # deg/pix
    xmin,xmax = -35,5
    nxpix = int((xmax-xmin)/pixscale) + 1
    ymin,ymax = -5,33
    nypix = int((ymax-ymin)/pixscale) + 1
    x = np.linspace(xmin,xmax,nxpix)
    y = np.linspace(ymin,ymax,nypix)
    XY = np.meshgrid(x,y)
    xx,yy = np.meshgrid(x,y)
    
    nside = hp.get_nside(hpxmap)
    pix = hp.ang2pix(nside,xx.flat,yy.flat,lonlat=True)

    val = hpxmap[pix]
    #mask = np.load('des/des_mask_sgr.npy')
    #m = mask[pix]
    #val[m==1] = np.nan
    #val[val<0] = np.nan
    #val[m==1] = np.nanmedian(val)
    vv = val.reshape(xx.shape)
    #smooth = smoothval.reshape(xx.shape)

    # Smoothed data
    smooth = nd.gaussian_filter(vv, sigma=sigma/pixscale)
    smooth1=np.copy(smooth)
    smooth1 = smooth1.reshape(val.shape)
    #mask = np.load('des/des_mask_sgr.npy')
    mask = np.load('pal13/decals_mask.npy')
    m = mask[pix]
    smooth1[m==1] = np.median(val) +1.5
    smooth1 = smooth1.reshape(xx.shape)

    deg = 5
    
    fit_params, fit_errors = curvefit2d(deg, XY, np.ravel(smooth1))
    
    #testing the limits of the background
    #fit_params = fit_params - fit_errors
    #fit_params = fit_params + fit_errors
    
    bkg = polynomial.polyval2d(xx, yy, fit_params)
    
    #plt.figure()
    #plt.title('data')
    #draw_image(xx,yy,smooth,**kwargs)

    #plt.figure()
    #plt.title('bkg')
    #draw_image(xx,yy,bkg,**kwargs)

    if dust is not None:
        #plt.figure(figsize=(6,6))
        #plt.title('dust')
        dustval = dust[pix].reshape(xx.shape)
        #kw = dict(vmin=0,vmax=0.2)
        #setdefaults(kw,kwargs)
        #plt.set_cmap('gray_r')
        #draw_image(xx,yy,dustval,**kw)

    #if planck is not None:
        #plt.figure(figsize=(6,6))
        #plt.title('planck')
        #pix = hp.ang2pix(hp.get_nside(planck),xx.flat,yy.flat,lonlat=True)
        #dustval = planck[pix].reshape(xx.shape)
        #kw = dict(vmin=0,vmax=250)
        #setdefaults(kw,kwargs)
        #plt.set_cmap('gray_r')
        #sdraw_image(xx,yy,dustval,**kw)
       
    #plt.figure(figsize=(6,6))
    #plt.title('background')
    #kw = dict(vmin=0.5,vmax=5)
    #setdefaults(kw,kwargs)
    #plt.set_cmap('gray_r')
    #draw_image(xx,yy,bkg,**kw)
    
    
    #plt.figure(figsize=(5,5))
    #plt.title('data - bkg')
    #plt.set_cmap('gray_r')
    #kwargs.update(vmin=-1,vmax=0.7)
    #draw_image(xx,yy,smooth-bkg,**kwargs)
    
    draw_all(xx,yy,smooth,dustval,bkg,**kwargs)
    
    fin = vv-bkg
    
    return xx,yy,fin
    
    
def stream_coord_spatial(hpxmap, ends, sigma=0.2, **kwargs):
    defaults = dict(cmap='gray_r', vmin=-2,vmax=2)
    setdefaults(kwargs, defaults)
    nside = 512
    phi1min, phi1max = -10, 10
    phi2min, phi2max = -5,5
    pixscale=0.1

    nphi1pix = int((phi1max-phi1min)/pixscale) + 1
    nphi2pix = int((phi2max-phi2min)/pixscale) + 1

    p1 = np.linspace(phi1min,phi1max,nphi1pix)
    p2 = np.linspace(phi2min,phi2max,nphi2pix)
    pp1,pp2 = np.meshgrid(p1,p2)

    phi1, phi2, R = streamlib.rotation(ends, np.array([0,1]), np.array([0,1]))
    
    #want to rotate backwards from phi1, phi2 to ra, dec
    ra, dec = streamlib.inv_rotation(pp1.flat[:], pp2.flat[:],R)
    pix = hp.ang2pix(nside,ra,dec,lonlat=True)
    
    val = hpxmap[pix]
        
    mask = np.load('pal13/Pal13_mask.npy')
    m = mask[pix]
    #val[m==1] = np.nan
    
    vv = val.reshape(pp1.shape)
    
    smooth = nd.gaussian_filter(vv, sigma=sigma/pixscale)
    
    smooth1 = smooth.reshape(val.shape)

    smooth1[m==1] = np.nan
    smooth = smooth1.reshape(pp1.shape)
    
    rem = np.where(np.isnan(smooth.flat))
    
    sm_for_bkg = np.delete(smooth.flat, rem)
    pp1_for_bkg = np.delete(pp1.flat, rem)
    pp2_for_bkg = np.delete(pp2.flat, rem)
    
    deg = 5
    coeff = polyfit2d(pp1_for_bkg, pp2_for_bkg, sm_for_bkg, [deg, deg])
    bkg = polynomial.polyval2d(pp1, pp2, coeff)
    
    plt.figure(figsize=(10,10))
    plt.title('data')
    plt.set_cmap('gray_r')
    #draw_image(xx,yy,bkg,**kwargs)
    kwargs.update(vmin=-2,vmax=1.5)
    draw_image(pp1,pp2,smooth-bkg, **kwargs)
    return pp1, pp2, vv
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    sfd = hp.read_map('lambda_sfd_ebv.fits')
    nside = hp.get_nside(sfd)
    ra,dec = hp.pix2ang(nside,np.arange(hp.nside2npix(nside)),lonlat=True)
    galpix = hp.ang2pix(nside,*cel2gal(ra,dec),lonlat=True)
    sfd = sfd[galpix]

    #planck = fitsio.read('COM_CompMap_ThermalDust-commander_2048_R2.00.fits')['I_ML_FULL']
    planck = hp.read_map('COM_CompMap_ThermalDust-commander_2048_R2.00.fits')
    nside = hp.get_nside(planck)
    ra,dec = hp.pix2ang(nside,np.arange(hp.nside2npix(nside)),lonlat=True)
    galpix = hp.ang2pix(nside,*cel2gal(ra,dec),lonlat=True)
    planck = planck[galpix]

    hpxcube, fracdet, modulus = load_hpxcube('test_hpxcube_pal13.fits.gz')
    print('modulus = {}'.format(modulus))
    mu = 16.8
    sigma=0.1
    vmin0, vmax0 = 0, 20
    #lon, lat = 20, -30
    lon, lat = 346, 12
    data = prepare_hpxmap(mu, hpxcube, fracdet, modulus)
    #bkg = 0
    #bkg = fit_background_poly(data)
    #smap = plot_density(data, bkg, center=(lon, lat), filename='pal13_density_{}.png'.format(int(10*mu)))
    xx, yy, smooth = run(data)
    draw_image(xx,yy,smooth)
