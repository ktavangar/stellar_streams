from __future__ import division, print_function

import os
import glob
import time
import traceback
import sys

import yaml
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

from ugali.analysis.isochrone import factory as isochrone_factory
from ugali.utils import healpix
from ugali.utils import fileio
from ugali.utils.projector import angsep

import mag_mag_lik_func as mmlf


def prior(model_parameters):
    # prior on parameter currently uniform
    # richness > 0
    # 10 < age < 14
    # 1e-4 < metallicity < 0.001
    richness=model_parameters[0]
    age=model_parameters[1]
    metallicity=mmlf.feh2z(-2.7)
    delta_mod=model_parameters[2]
    if richness <0 or age < 10 or age > 14 or metallicity < 3e-5 or metallicity > 0.01 or abs(delta_mod) > 2:
        return -np.inf
    else:
        return 0.0

def log_likelihood(model_parameters, binned_data, background_probabilities,
                   g_grid, r_grid, g_errors, r_errors):
    global COUNTER
    COUNTER += 1.
    if not COUNTER % int(0.01*TOTAL_SAMPLES):
        print(str(COUNTER)+"/"+str(TOTAL_SAMPLES)+":  "+str(np.round(COUNTER/TOTAL_SAMPLES,2)))
        sys.stdout.flush()

    if prior(model_parameters)==-np.inf:
        print('Prior not met')
        return(-np.inf)

    global AGE, METALLICITY, MODEL_PROBABILITIES
    if model_parameters[1] == AGE and model_parameters[2]==DELTA_MOD:
        print('Not updating')
        model_probabilities = MODEL_PROBABILITIES
    else:
        #print('Updating')
        try:
            model_probabilities = mmlf.bin_model(model_parameters, g_grid, r_grid, g_errors, r_errors, **config)
        except Exception:
            traceback.print_exc()
            print('problem!')
            print('model params = ', model_parameters)
            return -np.inf

        # update parameters
        #DISTANCE_MODULUS = model_parameters[1]
        AGE = model_parameters[1]
        MODEL_PROBABILITIES = model_probabilities

    richness = model_parameters[0]
    expected_counts = background_probabilities + richness * model_probabilities

    loglike = poisson.logpmf(binned_data, expected_counts)

    idx = (expected_counts == 0) & (binned_data != 0)

    gr_grid = g_grid - r_grid
    mask = (gr_grid > 0.0) & (gr_grid <= 1)

    # if priors are anything but uniform will need to include this bit.
    return np.sum(loglike[mask]) # + prior(model_parameters)


def simulate_stream(g_bins, r_bins, g_errors, r_errors, richness=2.e5, distance_modulus=16.25, age=12.8, metallicity=0.000034, scatter=0.1):
        # iso = isochrone_factory('Padova', distance_modulus=distance_modulus, age=age, z=metallicity)
        iso = isochrone_factory(ISO_TYPE, distance_modulus=distance_modulus, age=age, z=metallicity)

        # iso.mag_2 -= 0.08 # DELETE
        g_stream, r_stream = iso.simulate(richness * iso.stellar_mass())

        # scatter
        # magnitude_scatter = np.random.uniform(-scatter, scatter, size=(2, len(g_stream)))
        magnitude_scatter_g = np.random.normal(loc=0, scale=scatter, size=len(g_stream))  # for cleanest testing, use constant error
        magnitude_scatter_r = np.random.normal(loc=0, scale=scatter, size=len(r_stream))  # for cleanest testing, use constant error

        new_data = np.zeros((4, len(g_stream)))
        new_data[0] = g_stream + magnitude_scatter_g
        new_data[1] = r_stream + magnitude_scatter_r

        # ERRORS
        test = 0
        if test:
            # TEST
            new_data[2] = 2 * BIN_SIZE
            new_data[3] = 2 * BIN_SIZE
        else:
            # CORRECT
            g_error_spline = UnivariateSpline((g_bins[1:] + g_bins[:-1]) / 2, g_errors, s=0)
            r_error_spline = UnivariateSpline((r_bins[1:] + r_bins[:-1]) / 2, r_errors, s=0)

            new_data[2][(new_data[0] > G_MIN) & (new_data[0] < G_MAX)] = g_error_spline(new_data[0][(new_data[0] > G_MIN) & (new_data[0] < G_MAX)])
            new_data[3][(new_data[1] > R_MIN) & (new_data[1] < R_MAX)] = g_error_spline(new_data[1][(new_data[1] > R_MIN) & (new_data[1] < R_MAX)])
            new_data[2][new_data[2] == 0] = BIN_SIZE
            new_data[3][new_data[3] == 0] = BIN_SIZE

        return new_data


if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config')
    args = parser.parse_args()


    config = yaml.load(open(args.config))
#     TRUE_RICHNESS=args.richness
#     TRUE_AGE=args.age
#     TRUE_METALLICITY=args.metal
    MULTI=config["multi"]
    DISTANCE_MOD=config["DISTANCE_MOD"]
    DELTA_MOD=0
    RICHNESS = 0
    DISTANCE_MODULUS = 0
    AGE = 0
    METALLICITY = -2.7
    EBV = 0
    MODEL_PROBABILITIES = 0

    G_MIN = config["G_MIN"]
    G_MAX = config["G_MAX"]
    R_MIN = config["R_MIN"]
    R_MAX = config["R_MAX"]
    GR_MIN = config["GR_MIN"]
    GR_MAX = config["GR_MAX"]
    BIN_SIZE = config["BIN_SIZE"]
    ISO_TYPE = config["ISO_TYPE"]
    COUNTER=0

    #wscale=config["wscale"]
    #if wscale==1:
    #    zscale=0.15644016726198504
    #if wscale==2:
    #    zscale=0.31307426702873803
    #if wscale==3:
    #    zscale=0.4676384733287603#1
    zscale = 1
    AREA_SCALE=zscale


    g_bins, r_bins, g_grid, r_grid=mmlf.define_bins(g_min=G_MIN,
                                               g_max=G_MAX,
                                               r_min=R_MIN,
                                               r_max=R_MAX,
                                               bin_size=BIN_SIZE)
    print(config.keys())
    print(config['output_file'])
    offcat=np.load(config['offstream_cat'])
    background_cat=np.array([offcat['SOF_PSF_MAG_CORRECTED_G']-DISTANCE_MOD,offcat['SOF_PSF_MAG_CORRECTED_R']-DISTANCE_MOD,
                             offcat['SOF_PSF_MAG_ERR_G'],offcat['SOF_PSF_MAG_ERR_R']])

    oncat=np.load(config["onstream_cat"])
    stream_cat=np.array([oncat['SOF_PSF_MAG_CORRECTED_G']-DISTANCE_MOD,oncat['SOF_PSF_MAG_CORRECTED_R']-DISTANCE_MOD,
                         oncat['SOF_PSF_MAG_ERR_G'],oncat['SOF_PSF_MAG_ERR_R']])

    g_errors, r_errors=mmlf.propagate_errors(stream_cat, g_bins, r_bins, **config)

    binned_data=mmlf.bin_data(stream_cat,g_bins, r_bins)

    background_probabilities=mmlf.bin_background(background_cat,g_bins,r_bins, area_scale=AREA_SCALE)


    ndim=config["ndim"]
    nwalkers=config["nwalkers"]
    nsteps=int(config["nsteps"])

    TOTAL_SAMPLES=float(nsteps)


    from scipy.stats import truncnorm
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    X = get_truncated_normal(mean=12, sd=2, low=10, upp=14)
    r0=np.random.normal(loc=5e4, scale=1e4, size=nwalkers)
    a0=X.rvs(nwalkers)
    print(len(a0))
#     z0=np.random.normal(loc=0.0004,scale=0.0005,size=nwalkers)
    d0=np.random.normal(loc=0.0, scale=1, size=nwalkers)
    p0 = np.vstack([r0,a0,d0]).T


    #print(TRUE_RICHNESS,TRUE_AGE,TRUE_METALLICITY)

    print("STARTING SAMPLING")

    import pickle

    ################################################################
    # INITIALIZE EMCEE
    ################################################################

    if MULTI:
        print("multi")
        from multiprocessing import Pool
        pool=Pool(processes=config["multi"])
        #pool = MPIPool(loadbalance=True)
#         if not pool.is_master():
#             pool.wait()
#             sys.exit(0)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood,pool=pool,args=[binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors])

    else:
        print("single")

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=[binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors])

    ################################################################
    # RUN EMCEE
    ################################################################

    print("start")
    start = time.time()
    print(start)
    sys.stdout.flush()

    pos, prob, state = sampler.run_mcmc(p0, nsteps/10)
    sampler.reset()
    print('Starting real run')
    sampler.run_mcmc(pos, nsteps)
    end = time.time()
    serial_time = end - start

    ################################################################
    # SAVE RUN
    ################################################################
    print("SAMPLING DONE")
    print("Serial took {0:.1f} seconds".format(serial_time))
    samples = sampler.chain#[:, :, :].reshape((-1, ndim))
    with open(config["output_file"], 'wb') as f:
                   pickle.dump(samples, f)
    if MULTI:
        pool.close()
        
    for i in range(ndim):
        plt.figure()
        plt.hist(sampler.flatchain[:, i], 50, color="k", histtype="step")
        plt.title("Dimension {0:d}".format(i))
        #plt.savefig('likelihood_sgr_%i.png' % i)
    
    figure = corner.corner(sampler.flatchain, var_names=["rich", "age", "dm"])
    plt.show()

    samples = sampler.chain.reshape(-1, ndim)
    print(samples)
