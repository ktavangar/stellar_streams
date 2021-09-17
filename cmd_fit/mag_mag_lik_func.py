from __future__ import division, print_function

import os
import glob
import time
import traceback


import pandas as pd
import numpy as np
from collections import OrderedDict as odict
from matplotlib.colors import LogNorm
from matplotlib.path import Path

from scipy import stats
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.stats.distributions import poisson
import scipy.ndimage as nd

import matplotlib as mpl
import matplotlib.pyplot as plt

import emcee
from astropy.convolution import convolve, Gaussian1DKernel
import corner

# from ugali.isochrone import factory as isochrone_factory
from ugali.analysis.isochrone import factory as isochrone_factory
from ugali.utils import healpix
from ugali.utils import fileio
from ugali.utils.projector import angsep
#from schwimmbad import MPIPool


# setup function to bin data and background

def define_bins(g_min, g_max, r_min, r_max, bin_size):
    g_bins = np.arange(g_min, g_max + bin_size / 2, bin_size)
    r_bins = np.arange(r_min, r_max + bin_size / 2, bin_size)
    g_centers = (g_bins[1:] + g_bins[:-1]) / 2
    r_centers = (r_bins[1:] + r_bins[:-1]) / 2
    g_grid, r_grid = np.meshgrid(g_centers, r_centers)
    return g_bins, r_bins, g_grid.T, r_grid.T


def bin_data(data, g_bins, r_bins):
    g_mag, r_mag = data[:2]
    binned_data, _, _ = np.histogram2d(g_mag, r_mag, bins=[g_bins, r_bins], normed=False)
    return binned_data


def bin_data_cmd(data):
    gr_bins = np.arange(GR_MIN, GR_MAX + BIN_SIZE / 2, BIN_SIZE)
    g_bins = np.arange(G_MIN, G_MAX + BIN_SIZE / 2, BIN_SIZE)
    g_mag, r_mag = data[:2]
    binned_data, _, _ = np.histogram2d(g_mag - r_mag, g_mag, bins=[gr_bins, g_bins], normed=False)
    return binned_data.T


def propagate_errors(data, g_bins, r_bins,**kwargs):
    g_mag, r_mag, g_mag_err, r_mag_err = data

    g_errors, _, _ = stats.binned_statistic(g_mag, g_mag_err, statistic='median', bins=[g_bins])
    r_errors, _, _ = stats.binned_statistic(r_mag, r_mag_err, statistic='median', bins=[r_bins])

    minimum_error = kwargs["BIN_SIZE"]  # 0.01
    g_errors[np.isnan(g_errors)] = minimum_error
    r_errors[np.isnan(r_errors)] = minimum_error
    g_errors[g_errors < minimum_error] = minimum_error
    r_errors[r_errors < minimum_error] = minimum_error

    # g_errors, r_errors = np.meshgrid(g_errors, r_errors)
    return g_errors, r_errors  # temporary


def bin_background(background_data, g_bins, r_bins, area_scale=1):
    g_mag, r_mag, g_mag_err, r_mag_err = background_data

    background_probabilities, _, _ = np.histogram2d(g_mag, r_mag, bins=[g_bins, r_bins], normed=False)
    background_probabilities *= area_scale
    # set to min for now, try changing and see if affects results
    background_probabilities[background_probabilities == 0] = 1e-3  # CANNOT HAVE EXPECTED COUNTS = 0, ASTROPY CONVOLVE CUTS OFF, SO ADD NON-ZERO MIN BACKGROUND VALUE
    return background_probabilities


def simulate_background(background_probabilities):
    print('simulating background')
    background_probabilities = np.random.poisson(lam=background_probabilities, size=background_probabilities.shape)
    # background_probabilities = np.asarray(background_probabilities, dtype='float')
    # background_probabilities[background_probabilities == 0] = 1e-3
    return background_probabilities


def convolve_1d(probabilities, mag_err):
    sigma = mag_err / BIN_SIZE  # error in pixel units
    kernel = Gaussian1DKernel(sigma)
    convolved = convolve(probabilities, kernel)
    return convolved


def convolve_errors(probabilities, g_errors, r_errors):
    for i in range(len(g_errors)):
        probabilities[i] = convolve_1d(probabilities[i], g_errors[i])
    for i in range(len(r_errors)):
        probabilities[:, i] = convolve_1d(probabilities[:, i], r_errors[i])
        
    return probabilities



def feh2z(feh):
        # Section 3 of Dotter et al. 2008
        Y_p     = 0.245            # Primordial He abundance (WMAP, 2003)
        c       = 1.54             # He enrichment ratio 

        # This is not well defined...
        #Z_solar/X_solar = 0.0229  # Solar metal fraction (Grevesse 1998)
        ZX_solar = 0.0229
        return (1 - Y_p)/( (1 + c) + (1/ZX_solar) * 10**(-feh))
    
    
    
def bin_model(model_parameters, g_grid, r_grid, g_errors, r_errors, **kwargs):
    richness=model_parameters[0]
    age = model_parameters[1]
    metallicity = feh2z(-2.7)
    delta_mod=model_parameters[2]
    distance_mod=kwargs["DISTANCE_MOD"]
    BIN_SIZE=kwargs["BIN_SIZE"]
    
    iso = isochrone_factory(ISO_TYPE, distance_modulus=distance_mod, age=age, z=metallicity)
    # iso.mag_2 -= 0.08 # DELETE
    # print('SUBTRACTING 0.08')
    mass_init, mass_pdf, mass_act, mag_1, mag_2 = iso.sample()  # mass_min, mass_steps don't matter with renormalization
    mag_1 += delta_mod
    mag_2 += delta_mod
    mag_1_bins = np.arange(int(mag_1.min()), int(mag_1.max() + 1), BIN_SIZE)  # requires 1 be integer multiple of binsize
    mag_2_bins = np.arange(int(mag_2.min()), int(mag_2.max() + 1), BIN_SIZE)

    mag_1_centers = (mag_1_bins[1:] + mag_1_bins[:-1]) / 2
    mag_2_centers = (mag_2_bins[1:] + mag_2_bins[:-1]) / 2
    idx_1 = (mag_1_centers > G_MIN) & (mag_1_centers < G_MAX)
    idx_2 = (mag_2_centers > R_MIN) & (mag_2_centers < R_MAX)

    idx_3 = (g_grid[:, 0] >= mag_1_centers.min() - BIN_SIZE / 2) & (g_grid[:, 0] <= mag_1_centers.max() + BIN_SIZE / 2)
    idx_4 = (r_grid[0] >= mag_2_centers.min() - BIN_SIZE / 2) & (r_grid[0] <= mag_2_centers.max() + BIN_SIZE / 2)

    full_probabilities, _, _ = np.histogram2d(mag_1, mag_2, bins=[mag_1_bins, mag_2_bins], weights=mass_pdf, normed=True)
    full_probabilities /= (full_probabilities.sum())
    
    model_probabilities = np.zeros_like(g_grid)
    model_probabilities[np.ix_(idx_3, idx_4)] = full_probabilities[np.ix_(idx_1, idx_2)]
    

    # convolve errors
    norm = model_probabilities.sum()
    model_probabilities = convolve_errors(model_probabilities, g_errors[:], r_errors[:])
    model_probabilities *= (norm / model_probabilities.sum())  # keep same normalization

    return model_probabilities

G_MIN = -0.2#16.
G_MAX = 7.5  # 5
R_MIN = -0.2
R_MAX = 7.5  # 5
GR_MIN = 0.0
GR_MAX = 1.0
BIN_SIZE = 0.05
ISO_TYPE = 'Dotter2008'
