import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/data/des81.b/data/tavangar/streams/code')
import glob

import numpy as np
import healpy as hp
import fitsio as fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage as nd
from utils import load_infiles
import pandas as pd

import ugali
from ugali import isochrone

import skymap, skymap.survey
from skymap.utils import cel2gal, gal2cel
from skymap.utils import setdefaults

from streamlib import skymap_factory
import streamlib
import results
import rotation_matrix
import plot_hess
import filter_data

import polyfit2d
from numpy.polynomial import polynomial

import importlib
import imp

import region_plot

#load the des data to create the isochrones
from joblib import parallel_backend

ra, dec = 24, -50
ras = []
decs = []
for i in range(20):
    ras = np.append(ras, ra+i)
    ras = np.append(ras, ra-i)
for i in range(20):
    decs = np.append(decs, dec+i)
    decs = np.append(decs, dec-i)
    
def ang2pix(nside, lon, lat, nest=False):
    """
    Input (lon, lat) in degrees instead of (theta, phi) in radians
    """
    pix = []
    theta = np.radians(90. - lat)
    phi = np.radians(lon)
    for i in range(len(theta)):
        pix = np.append(pix,hp.ang2pix(nside, theta[i], phi, nest=nest))
    print(len(pix))
    pix = np.unique(pix)
    return pix
ind = ang2pix(32, ras, decs)
print(ind)
print(len(ind))
filenames = []
for i in range(len(ind)):
    if ind[i] < 10000:
        filenames = np.append(filenames, '/data/des40.b/data/gaia/edr3/healpix/GaiaSource_0{}.fits'.format(int(ind[i])))
    else:
        filenames = np.append(filenames, '/data/des40.b/data/gaia/edr3/healpix/GaiaSource_{}.fits'.format(int(ind[i])))

print(filenames)

with parallel_backend('threading'):
    gaia_data = load_infiles(filenames,columns=[
        'RA','RA_ERROR','DEC','DEC_ERROR','PMRA','PMRA_ERROR','PMDEC','PMDEC_ERROR',
        'PHOT_G_MEAN_MAG','PHOT_BP_MEAN_MAG', 'PHOT_RP_MEAN_MAG','BP_RP','BP_G','G_RP'])
np.save('gaia_phoenix_data.npy', gaia_data)
