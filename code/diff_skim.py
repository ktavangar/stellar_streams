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

from polyfit2d import polyfit2d
from numpy.polynomial import polynomial

import importlib
import imp

import region_plot

filenames = glob.glob('/home/s1/kadrlica/projects/y3a2/data/gold/v2.0/healpix/*.fits')
full_data = load_infiles(filenames,columns=[
    'RA','DEC','SOF_PSF_MAG_G','SOF_PSF_MAG_R','SOF_PSF_MAG_I', 'EXTENDED_CLASS_MASH_SOF', 
    'SPREAD_MODEL_G', 'SPREADERR_MODEL_G', 'SPREAD_MODEL_R', 'SPREADERR_MODEL_R'],multiproc=8)

smg = full_data['SPREAD_MODEL_G']
smgerr = full_data['SPREADERR_MODEL_G']
smr = full_data['SPREAD_MODEL_R']
smrerr = full_data['SPREADERR_MODEL_R']
smgerr2 = smgerr**2
smrerr2 = smrerr**2
full_data = full_data[(np.abs(smg/smgerr2 + smr/smrerr2) * (1/smgerr2 + 1/smrerr2)**-1) < 0.003]
g,r,i=full_data['SOF_PSF_MAG_G'], full_data['SOF_PSF_MAG_R'], full_data['SOF_PSF_MAG_I']
full_data = full_data[(np.abs(r-i-0.04-0.4*(g-r-0.25))) < 0.1]
print(len(full_data))

data = np.copy(full_data)
np.save('other_skim.npy', data)
