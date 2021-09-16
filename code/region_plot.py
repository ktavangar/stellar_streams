#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Kiyan Tavangar"

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

import polyfit2d
from numpy.polynomial import polynomial

from scipy.optimize import curve_fit

RA1,DEC1 = 31, -33
RA2,DEC2 = 13,-23
#RA,DEC = 78.52, -40.05
#RA -= 360*(RA>180)


def plot_pretty(dpi=175, fontsize=15, labelsize=20, figsize=(10, 8), tex=True):
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
    mpl.rcParams.update({'figure.autolayout': False})



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

def fit_bkg_poly(data, sigma=0.1, percent=[2, 95], deg=5):
    """ Fit foreground/background with a polynomial """
    nside = hp.get_nside(data)
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    lon -= 360*(lon > 180)

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
    bkg = np.ma.array(bkg, fill_value=np.nan)
    return bkg

def draw_image(xx,yy,data,**kwargs):
    #kwargs.update(vmin=-5,vmax=5)
    ax = plt.gca()
    ax.pcolormesh(xx,yy,data,rasterized=True, **kwargs)
    #ax.plot(lon1,lat1, c='b', linewidth=1.5, linestyle='dashed')
    ax.set_aspect('equal')
    plt.grid('on',ls=':',color='gray')
    ax.invert_xaxis()
    #ax.set_ylabel('Dec (deg)', fontsize=15)
    #ax.set_xlabel('RA (deg)', fontsize=15)

def make_mask(xx,yy,ra,dec,halfheight, halfwidth):
    remx = np.where(((ra-halfwidth) < xx.flat) & (xx.flat < (ra + halfwidth)))
    remy = np.where(((dec-halfheight) < yy.flat) & (yy.flat < (dec + halfheight)))
    rem = np.intersect1d(remx, remy)
    return rem

smap=skymap_factory('pal13')
lon1,lat1 = smap().draw_great_circle(-40.7,-59.9, -28.3, -43, 'full')
    
def run(hpxmap, dust=None, planck=None, sigma=0.25,**kwargs):
    defaults = dict(cmap='gray_r')
    setdefaults(kwargs, defaults)

    pixscale = 0.1 # deg/pix
    xmin,xmax = 150,180
    #xmin,xmax = -30,0
    xmin,xmax = 10,40
    nxpix = int((xmax-xmin)/pixscale) + 1
    ymin,ymax = -50,-10
    #ymin,ymax = -1.7,1.8
    #ymin,ymax = -15,20
    #ymin,ymax = -64,-30 # EriPhe overdensity
    
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
    smooth = smooth.reshape(val.shape)
    smooth1 = smooth1.reshape(val.shape)
    mask = np.load('des/des_mask_sgr.npy')
    #mask = np.load('pal13/decals_mask.npy')
    m = mask[pix]
    smooth[m==1] = np.nan
    smooth[smooth>6] = np.nan
    smooth = smooth.reshape(xx.shape)
    smooth1[m==1] = np.median(val) + 0.5
    smooth1 = smooth1.reshape(xx.shape)

    deg = 5
    
    fit_params, fit_errors = polyfit2d.curvefit2d(deg, XY, np.ravel(smooth1))
    
    #testing the limits of the background
    #fit_params = fit_params - fit_errors
    #fit_params = fit_params + fit_errors
    
    bkg = polynomial.polyval2d(xx, yy, fit_params)
    
    '''
    npix = hp.nside2npix(512)
    bkgmap = np.zeros(npix)
    ras, decs = hp.pix2ang(512, np.arange(npix), lonlat=True)
    for i in range(len(ras)):
        if ras[i] >300:
            ras[i] = ras[i] - 360
    pix2 = []
    pix2 = np.append(pix2, np.where((ras > -35) & (ras < 5) & (decs > -5) & (decs < 33)))
    print(pix2)
                    
    for i in pix2:
        ra, dec = hp.pix2ang(512, int(i), lonlat=True)
        print(ra, dec)
        if ra>300:
            ra = ra-360
        value = polynomial.polyval2d(ra, dec, fit_params)
        bkgmap[int(i)] = value
    #hp.mollview(bkgmap, min = 0, max=5)
    np.save('bkgmap_pal13.npy', bkgmap)
    '''

    #plt.figure()
    #plt.title('data')
    #draw_image(xx,yy,smooth,**kwargs)

    #plt.figure()
    #plt.title('bkg')
    #draw_image(xx,yy,bkg,**kwargs)

    '''
    if dust is not None:
        plt.figure(figsize=(8,4))
        plt.title('dust')
        dustval = dust[pix].reshape(xx.shape)
        kw = dict(vmin=0,vmax=0.2)
        setdefaults(kw,kwargs)
        draw_image(xx,yy,dustval,**kw)

    if planck is not None:
        plt.figure(figsize=(8,4))
        plt.title('planck')
        pix = hp.ang2pix(hp.get_nside(planck),xx.flat,yy.flat,lonlat=True)
        dustval = planck[pix].reshape(xx.shape)
        kw = dict(vmin=0,vmax=250)
        setdefaults(kw,kwargs)
        draw_image(xx,yy,dustval,**kw)
    '''  

    plt.figure(figsize=(10,10))
    plt.title('data - bkg')
    plt.set_cmap('gray_r')
    kwargs.update(vmin=2,vmax=5)
    #kwargs.update(vmin=0,vmax=12)
    #draw_image(xx,yy,smooth,**kwargs)

    draw_image(xx,yy,smooth,**kwargs)
    res = smooth-bkg
    return xx,yy,res

def bkg_testing(hpxmap, dust=None, planck=None, sigma=0.2,**kwargs):
    defaults = dict(cmap='gray_r')
    setdefaults(kwargs, defaults)

    pixscale = 0.1 # deg/pix
    xmin,xmax = -51, -10
    nxpix = int((xmax-xmin)/pixscale) + 1
    ymin,ymax = -65,-41
    nypix = int((ymax-ymin)/pixscale) + 1
    x = np.linspace(xmin,xmax,nxpix)
    y = np.linspace(ymin,ymax,nypix)
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

    #rem = make_mask(xx,yy,40,-34.5,2, 2)
    #vv.flat[rem]=4
    #rem = make_mask(xx,yy,24.5,-57.5,2, 3)
    #vv.flat[rem] = 7
    #rem = make_mask(xx,yy,55,-35.5,0.5, 1.5)
    #vv.flat[rem] = 4
    #rem = make_mask(xx,yy,54.5,-23,0.5, 0.5)
    #vv.flat[rem] = 4
    #rem = make_mask(xx,yy,54.5,-23,0.5, 0.5)
    #vv.flat[rem] = 4
    #rem = make_mask(xx,yy,77,-65,11, 20)
    #vv.flat[rem] = 4
    # Smoothed data
    smooth = nd.gaussian_filter(vv, sigma=sigma/pixscale)
    smooth1 = smooth.reshape(val.shape)
    mask = np.load('des/des_mask_sgr.npy')
    m = mask[pix]
    smooth1[m==1] = np.nan
    smooth = smooth1.reshape(xx.shape)
    rem = np.where(np.isnan(smooth.flat))
    
    sm_for_bkg = np.delete(smooth.flat, rem)
    xx_for_bkg = np.delete(xx.flat, rem)
    yy_for_bkg = np.delete(yy.flat, rem)
    
    deg = 5
    coeff = polyfit2d(xx_for_bkg, yy_for_bkg, sm_for_bkg, [deg, deg])
    bkg = polynomial.polyval2d(xx, yy, coeff)
    #smoothflat = np.ma.array(smooth.flat, mask=np.isnan(smooth))
    #deg = 5
    #coeff = polyfit2d(xx.flat, yy.flat, smooth.flat, [deg, deg])
    #bkg = polynomial.polyval2d(xx, yy, coeff)
    
    #bkg=fit_bkg_poly(vv.flat, sigma=0.2, percent=[2, 95], deg=5)
    #bkg=0

    #plt.figure()
    #plt.title('data')
    #draw_image(xx,yy,smooth,**kwargs)

    #plt.figure()
    #plt.title('bkg')
    #draw_image(xx,yy,bkg,**kwargs)

    '''
    if dust is not None:
        plt.figure(figsize=(8,4))
        plt.title('dust')
        dustval = dust[pix].reshape(xx.shape)
        kw = dict(vmin=0,vmax=0.2)
        setdefaults(kw,kwargs)
        draw_image(xx,yy,dustval,**kw)

    if planck is not None:
        plt.figure(figsize=(8,4))
        plt.title('planck')
        pix = hp.ang2pix(hp.get_nside(planck),xx.flat,yy.flat,lonlat=True)
        dustval = planck[pix].reshape(xx.shape)
        kw = dict(vmin=0,vmax=250)
        setdefaults(kw,kwargs)
        draw_image(xx,yy,dustval,**kw)
    '''   

    plt.figure(figsize=(10,10))
    plt.title('data - bkg')
    plt.set_cmap('gray_r')
    kwargs.update(vmin=-2,vmax=2)
    #draw_image(xx,yy,bkg,**kwargs)

    draw_image(xx,yy,smooth,**kwargs)
    
    return xx,yy,smooth

def run_atlas(hpxmap, dust=None, planck=None, sigma=0.2,**kwargs):
    defaults = dict(cmap='gray_r')
    setdefaults(kwargs, defaults)

    pixscale = 0.1 # deg/pix
    xmin,xmax = 5,40
    nxpix = int((xmax-xmin)/pixscale) + 1
    ymin,ymax = -40,-15
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
    smooth1 = smooth.reshape(val.shape)
    mask = np.load('des/des_mask_no_sgr.npy')
    m = mask[pix]
    smooth2 = np.copy(smooth1)
    smooth1[m==1] = np.median(val)
    smooth2[m==1] = np.nan
    smooth = smooth2.reshape(xx.shape)
    
    deg = 5
    
    fit_params, fit_errors = polyfit2d.curvefit2d(deg, XY, np.ravel(smooth1))
    
    #testing the limits of the background
    #fit_params = fit_params - fit_errors
    #fit_params = fit_params + fit_errors
    
    bkg = polynomial.polyval2d(xx, yy, fit_params)
    
    #bkg=fit_bkg_poly(vv.flat, sigma=0.2, percent=[2, 95], deg=5)
    #bkg=0

    #plt.figure()
    #plt.title('data')
    #draw_image(xx,yy,smooth,**kwargs)

    #plt.figure()
    #plt.title('bkg')
    #draw_image(xx,yy,bkg,**kwargs)

    '''
    if dust is not None:
        plt.figure(figsize=(8,4))
        plt.title('dust')
        dustval = dust[pix].reshape(xx.shape)
        kw = dict(vmin=0,vmax=0.2)
        setdefaults(kw,kwargs)
        draw_image(xx,yy,dustval,**kw)

    if planck is not None:
        plt.figure(figsize=(8,4))
        plt.title('planck')
        pix = hp.ang2pix(hp.get_nside(planck),xx.flat,yy.flat,lonlat=True)
        dustval = planck[pix].reshape(xx.shape)
        kw = dict(vmin=0,vmax=250)
        setdefaults(kw,kwargs)
        draw_image(xx,yy,dustval,**kw)
    '''   

    plt.figure(figsize=(10,10))
    plt.title('data - bkg')
    plt.set_cmap('gray_r')
    kwargs.update(vmin=-2,vmax=2)
    #draw_image(xx,yy,smooth,**kwargs)

    draw_image(xx,yy,smooth-bkg,**kwargs)
    
    return xx,yy,smooth

def run_phoenix(hpxmap, dust=None, planck=None, sigma=0.2,**kwargs):
    defaults = dict(cmap='gray_r')
    setdefaults(kwargs, defaults)

    pixscale = 0.1 # deg/pix
    xmin,xmax = 5,45
    nxpix = int((xmax-xmin)/pixscale) + 1
    ymin,ymax = -62,-33
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
    smooth1 = smooth.reshape(val.shape)
    mask = np.load('des/des_mask_no_sgr.npy')
    m = mask[pix]
    smooth2 = np.copy(smooth1)
    smooth1[m==1] = np.median(val)
    smooth2[m==1] = np.nan
    smooth = smooth2.reshape(xx.shape)
    
    deg = 5
    
    fit_params, fit_errors = polyfit2d.curvefit2d(deg, XY, np.ravel(smooth1))
    
    #testing the limits of the background
    #fit_params = fit_params - fit_errors
    #fit_params = fit_params + fit_errors
    
    bkg = polynomial.polyval2d(xx, yy, fit_params)
    
    #bkg=fit_bkg_poly(vv.flat, sigma=0.2, percent=[2, 95], deg=5)
    #bkg=0

    #plt.figure()
    #plt.title('data')
    #draw_image(xx,yy,smooth,**kwargs)

    #plt.figure()
    #plt.title('bkg')
    #draw_image(xx,yy,bkg,**kwargs)

    '''
    if dust is not None:
        plt.figure(figsize=(8,4))
        plt.title('dust')
        dustval = dust[pix].reshape(xx.shape)
        kw = dict(vmin=0,vmax=0.2)
        setdefaults(kw,kwargs)
        draw_image(xx,yy,dustval,**kw)
    
    if planck is not None:
        plt.figure(figsize=(8,4))
        plt.title('planck')
        pix = hp.ang2pix(hp.get_nside(planck),xx.flat,yy.flat,lonlat=True)
        dustval = planck[pix].reshape(xx.shape)
        kw = dict(vmin=0,vmax=250)
        setdefaults(kw,kwargs)
        draw_image(xx,yy,dustval,**kw)
    '''   
    plot_pretty(fontsize=20)
    plt.title('data - bkg')
    plt.set_cmap('gray_r')
    kwargs.update(vmin=-1.5,vmax=1.5)
    #draw_image(xx,yy,smooth,**kwargs)

    draw_image(xx,yy,smooth-bkg,**kwargs)
    
    return xx,yy,smooth-bkg

def stream_coord_spatial(hpxmap, ends, data, deg, sigma=0.2, stream='Pal13', **kwargs):
    defaults = dict(cmap='gray_r', vmin=0,vmax=5)
    setdefaults(kwargs, defaults)
    nside = 512
    if stream == 'Phoenix':
        phi1min, phi1max = -9, 7.4
    if stream == 'Atlas':
        phi1min, phi1max = -10, 13
    if stream == 'Palca':
        phi1min, phi1max = -10, 10
    #phi1min, phi1max = -6, 7
    phi2min, phi2max = -6, 6
    pixscale = 0.1
    pixscale1=0.2
    pixscale2=0.05

    #nphi1pix = int((phi1max-phi1min)/pixscale) + 1
    nphi1pix = (phi1max-phi1min)/pixscale1 + 2 # for a 0.2 x 0.05 pixel size
    #nphi2pix = int((phi2max-phi2min)/pixscale) + 1
    nphi2pix = (phi2max-phi2min)/pixscale2 + 1 # for a 0.2 x 0.05 pixel size

    p1 = np.linspace(phi1min,phi1max,nphi1pix)
    p2 = np.linspace(phi2min,phi2max,nphi2pix)
    PP = np.meshgrid(p1,p2)
    pp1,pp2 = np.meshgrid(p1,p2)

    phi1, phi2, R = streamlib.rotation(ends, data['RA'], data['DEC'])
    if stream=='Pal13':
        cluster_rot_phi1, cluster_rot_phi2, R = streamlib.rotation(ends, np.array([346.685333]), np.array([12.770972]))
    
    #want to rotate backwards from phi1, phi2 to ra, dec
    ra, dec = streamlib.inv_rotation(pp1.flat[:], pp2.flat[:],R)
    pix = hp.ang2pix(nside,ra,dec,lonlat=True)
    
    val = hpxmap[pix]
    val_mask = hpxmap[pix]
    #print(val_mask)
        
    mask = np.load('des/des_mask_no_sgr.npy')
    #mask = np.load('pal13/decals_mask.npy')
    m = mask[pix]
    print(np.median(val))
    
    #val_mask[m==1] = np.nan #phoenix
    val_mask[m==1] = np.median(val)+0.5
    #print(np.median(val)+0.5)
    
    vv = val_mask.reshape(pp1.shape)
    vv_mask = val_mask.reshape(pp1.shape)
    
    #marea = np.where((pp1.flat<-6) & (vv_mask.flat==0))
    #vv_mask.flat[marea] = np.median(val)+1.5
    
    smooth = nd.gaussian_filter(vv_mask, sigma=sigma/pixscale)
    smooth1 = np.copy(smooth)
    smooth1 = smooth1.reshape(val.shape)

    smooth1[m==1] = np.nan
    #smooth = smooth1.reshape(pp1.shape)
    smooth1 = smooth1.reshape(pp1.shape)
    
    #curve_fit method to get uncertainties:
    fit_params, fit_errors = polyfit2d.curvefit2d(deg, PP, np.ravel(smooth))
    
    #deviation = np.multiply(fit_errors, np.array(np.random.normal(0,1,[deg+1,deg+1])))
    #fit_params = fit_params + deviation
    #print(fit_params)
    
    #testing the limits of the background
    #fit_params = fit_params - fit_errors
    #fit_params = fit_params + fit_errors
    
    bkg = polynomial.polyval2d(pp1, pp2, fit_params)
    
    #smooth = smooth[80:160] # these are currently set for +- 2 degrees
    #smooth1 = smooth1[80:160]
    #bkg = bkg[80:160]
    #pp1 = pp1[80:160]
    #pp2 = pp2[80:160]
    #vv = vv[80:160]
    #vv_mask = vv_mask[80:160]
    
    #smooth = smooth[40:200] # these are currently set for +- 4 degrees
    #smooth1 = smooth1[40:200]
    #bkg = bkg[40:200]
    #pp1 = pp1[40:200]
    #pp2 = pp2[40:200]
    #vv = vv[40:200]
    #vv_mask = vv_mask[40:200]
    
    #smooth = smooth[:, :43] # use :43 and 42: to split into left and right
    #smooth1 = smooth1[:, :43]
    #bkg = bkg[:, :43]
    #pp1 = pp1[:, :43]
    #pp2 = pp2[:, :43]
    #vv = vv[:, :43]
    #vv_mask = vv_mask[:, :43]

    
    #plt.figure(figsize=(10,10))
    #kwargs.update(vmin=0,vmax=5)
    #draw_image(pp1,pp2,smooth1, **kwargs)
    #plt.figure(figsize=(10,10))
    #kwargs.update(vmin=0,vmax=5)
    #draw_image(pp1,pp2,bkg, **kwargs)
    
    plt.figure(figsize=(10,10))
    #plt.title('data')
    plt.set_cmap('gray_r')
    #draw_image(xx,yy,bkg,**kwargs)
    kwargs.update(vmin=-1.5,vmax=1.5)
    draw_image(pp1,pp2,smooth-bkg, **kwargs)
    return pp1, pp2, vv, smooth-bkg, smooth, vv_mask, vv-bkg


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
