from __future__ import division, print_function

import os
import glob
import time

import numpy as np
from collections import OrderedDict as odict

from scipy import stats
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.stats.distributions import poisson
import scipy.ndimage as nd

import matplotlib as mpl
import matplotlib.pyplot as plt

import emcee
from astropy.convolution import convolve, Gaussian1DKernel

# from ugali.isochrone import factory as isochrone_factory
from ugali.analysis.isochrone import factory as isochrone_factory
from ugali.utils import healpix
from ugali.utils import fileio
from ugali.utils.projector import angsep

# r, mu, age, z, n_background
STREAM_PARAMS = odict([
    ('ATLAS', (10000, 16.5, 12.5, 0.0005, 2.)),
    ('Phoenix', (10000, 16.21, 11.5, 0.0004, 2.)),
    ('TucIII', (10000, 17., 10.9, 0.0001, 2.)),
    ('Vertical', (10000, 17., 12., 0.0001, 1.)),
    ('New', (10000, 16., 12.5, 0.0001, 2.)),
    ('Sagittarius', (3750000, 16.95, 10.47, 0.0007, 1.))
])


# STREAM = 'TucIII' # reformat this eventually, doesn't matter for now
# TRUE_RICHNESS, TRUE_DISTANCE_MODULUS, TRUE_AGE, TRUE_METALLICITY, N_BACKGROUND = STREAM_PARAMS[STREAM]

# M2
STREAM = 'M2'
RA, DEC = 323.362552 - 360., -0.823318
RADIUS = 0.1
N_BACKGROUND = 2
TRUE_RICHNESS, TRUE_DISTANCE_MODULUS, TRUE_AGE, TRUE_METALLICITY = 1.e4, 15.48, 12.5, 0.00039

RICHNESS = 0
DISTANCE_MODULUS = 0
AGE = 0
METALLICITY = 0
MODEL_PROBABILITIES = 0

"""
TRUE_RICHNESS = 3750000  # Sgr
TRUE_DISTANCE_MODULUS = 16.95  # 16.
TRUE_AGE = 10.47  # 12.
TRUE_METALLICITY = 0.0007  # 0.0004
N_BACKGROUND = 2.
"""

G_MIN = 16.
G_MAX = 22.
R_MIN = 16.
R_MAX = 22.
BIN_SIZE = 0.05

COUNTER = 0.

DATA_DIR = '/home/s1/nshipp/projects/field_of_streams/data/likelihood_y3a2/'
ISO_DIR = '/home/s1/nshipp/.ugali/isochrones/padova' # old
# ISO_DIR = '/home/s1/nshipp/.ugali/isochrones/des/bressan2012' # new
# ISO_DIR = 'home/s1/nshipp/.ugali/isochrones/des/dotter2016' # new, dotter
# ISO_DIR = 'home/s1/nshipp/.ugali/isochrones/dotter' # old, dotter


def load_data2(ra, dec, radius=0.3):
    filename = '/data/des40.b/data/y3a2/gold/v1.0/healpix/y3_gold_1_0_%05d.fits'
    columns = ['RA', 'DEC', 'WAVG_MAG_PSF_G', 'WAVG_MAG_PSF_R', 'EBV_SFD98', 'WAVG_SPREAD_MODEL_R', 'WAVG_MAGERR_PSF_G', 'WAVG_MAGERR_PSF_R']
    pix = healpix.ang2disc(32, ra, dec, 2 * radius, inclusive=True)

    files = []
    for p in pix:
        # print 'pixel: ', p
        f = filename % p
        if os.path.exists(f):
            files.append(f)

    data = fileio.load_infiles(files, columns=columns)
    sel = (angsep(ra, dec, data['RA'], data['DEC']) < radius)
    sel &= data['WAVG_SPREAD_MODEL_R'] < 0.003
    d = data[sel]

    g = d['WAVG_MAG_PSF_G'] - 3.186 * (d['EBV_SFD98'] + 0.013)
    r = d['WAVG_MAG_PSF_R'] - 2.140 * (d['EBV_SFD98'] + 0.013)
    g_err = d['WAVG_MAGERR_PSF_G']
    r_err = d['WAVG_MAGERR_PSF_R']

    return np.vstack([g, r, g_err, r_err])


def load_data(stream):
    cols = [2, 3, 4, 5]  # g, r, g_err, r_err (not loading ra, dec)
    filename = DATA_DIR + '%s.txt' % stream
    print('data file = %s' % filename)
    data = np.loadtxt(filename, delimiter=',', unpack=True, usecols=cols)
    background_data = np.loadtxt(DATA_DIR + '%s_background.txt' % stream, delimiter=',', unpack=True, usecols=cols)
    print(len(data[0]), len(background_data[0]) / N_BACKGROUND)
    # data[0] -= 3.186 * 0.013
    # data[1] -= 2.140 * 0.013
    # background_data[0] -= 3.186 * 0.013
    # background_data[1] -= 2.140 * 0.013
    return data, background_data


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
    # binned_data[binned_data == 0] = 1  # set to min, integer required for poisson
    return binned_data

''''
def bin_data_gr(data):
    g_min = G_MIN
    g_max = G_MAX
    bin_size = bin_size

    gr_min = -0.5
    gr_max = 1.5
    gr_bin_size = 0.05

    g_bins = np.arange(g_ming, g_max + bin_size / 2, bin_size)
    r_bins = np.arange(r_min, r_max + bin_size / 2, bin_size)
    g_mag, r_mag = data[:2]
    binned_data, _, _ = np.histogram2d(g_mag, r_mag, bins=[g_bins, r_bins], normed=False)
    # binned_data[binned_data == 0] = 1  # set to min, integer required for poisson
    return binned_data
'''


def propagate_errors(data, g_bins, r_bins):
    g_mag, r_mag, g_mag_err, r_mag_err = data

    g_errors, _, _ = stats.binned_statistic(g_mag, g_mag_err, statistic='median', bins=[g_bins])
    r_errors, _, _ = stats.binned_statistic(r_mag, r_mag_err, statistic='median', bins=[r_bins])

    minimum_error = BIN_SIZE  # 0.01
    g_errors[np.isnan(g_errors)] = minimum_error
    r_errors[np.isnan(r_errors)] = minimum_error
    g_errors[g_errors < minimum_error] = minimum_error
    r_errors[r_errors < minimum_error] = minimum_error

    # g_errors, r_errors = np.meshgrid(g_errors, r_errors)
    return g_errors, r_errors  # temporary


def bin_background(background_data, g_bins, r_bins):
    g_mag, r_mag, g_mag_err, r_mag_err = background_data

    background_probabilities, _, _ = np.histogram2d(g_mag, r_mag, bins=[g_bins, r_bins], normed=False)
    background_probabilities /= N_BACKGROUND  # when using N x area (phoenix_background_N.txt)
    # background_probabilities = background_probabilities.astype(int)

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

    # print('acceptable min?', probabilities[probabilities > 0].min()) ~1e-13 or lower
    # probabilities[probabilities == 0] = probabilities[probabilities > 0].min() # setting background to min instead
    return probabilities


def calculate_observable_fraction(iso, mass_min=0.1):
    # mass_min = 0.1 ??
    mass_init, mass_pdf, mass_act, mag_1, mag_2 = iso.sample(mass_min=mass_min, full_data_range=False)

    g_cut = (mag_1 + iso.distance_modulus < G_MAX) & (mag_1 + iso.distance_modulus > G_MIN)
    r_cut = (mag_2 + iso.distance_modulus < R_MAX) & (mag_2 + iso.distance_modulus > R_MIN)

    mass_pdf_cut = mass_pdf * g_cut * r_cut
    observable_fraction = np.sum(mass_pdf_cut)  # / np.sum(mass_pdf)
    return observable_fraction


def bin_model(model_parameters, g_grid, r_grid, g_errors, r_errors):
    distance_modulus = model_parameters[1]
    age = model_parameters[2]
    metallicity = model_parameters[3]

    # iso = isochrone_factory('Padova', distance_modulus=distance_modulus, age=age, z=metallicity)
    iso = isochrone_factory('Padova', distance_modulus=distance_modulus, age=age, z=metallicity, dirname=ISO_DIR)
    # iso.mag_2 -= 0.08 # DELETE
    # print('SUBTRACTING 0.08')
    mass_init, mass_pdf, mass_act, mag_1, mag_2 = iso.sample()  # mass_min, mass_steps don't matter with renormalization

    mag_1 += iso.distance_modulus
    mag_2 += iso.distance_modulus
    mag_1_bins = np.arange(int(mag_1.min()), int(mag_1.max() + 1), BIN_SIZE)  # requires 1 be integer multiple of binsize
    mag_2_bins = np.arange(int(mag_2.min()), int(mag_2.max() + 1), BIN_SIZE)

    mag_1_centers = (mag_1_bins[1:] + mag_1_bins[:-1]) / 2
    mag_2_centers = (mag_2_bins[1:] + mag_2_bins[:-1]) / 2
    idx_1 = (mag_1_centers > G_MIN) & (mag_1_centers < G_MAX)
    idx_2 = (mag_2_centers > R_MIN) & (mag_2_centers < R_MAX)

    idx_3 = (g_grid[:, 0] >= mag_1_centers.min() - BIN_SIZE / 2) & (g_grid[:, 0] <= mag_1_centers.max() + BIN_SIZE / 2)
    idx_4 = (r_grid[0] >= mag_2_centers.min() - BIN_SIZE / 2) & (r_grid[0] <= mag_2_centers.max() + BIN_SIZE / 2)

    full_probabilities, _, _ = np.histogram2d(mag_1, mag_2, bins=[mag_1_bins, mag_2_bins], weights=mass_pdf, normed=True)
    full_probabilities /= (full_probabilities.sum())  # normalizing means INTEGRATING gives 1
    # full_probabilities /= (full_probabilities.sum() * BIN_SIZE**2)  # normalizing means INTEGRATING gives 1

    # print('full probs norm = %.8f' % full_probabilities.sum())
    # has to be normed to multiply by binsize to integrate, but if not normed then sum of model probabilities (without errors) equals correct number of observable stars

    # this indexing doesn't work when G_MIN lower than mag_1, etc. (for certain distance moduli)
    # g_error_array = np.full(full_probabilities.shape[0], 1e-30)
    # r_error_array = np.full(full_probabilities.shape[1], 1e-30)
    # g_error_array[idx_1] = g_errors[0]
    # r_error_array[idx_2] = r_errors[:, 0]

    # full_probabilities = convolve_errors(full_probabilities, g_error_array, r_error_array)

    model_probabilities = np.zeros_like(g_grid)
    model_probabilities[np.ix_(idx_3, idx_4)] = full_probabilities[np.ix_(idx_1, idx_2)]
    # model_probabilities *= BIN_SIZE**2
    # print('model probs norm = %.8f' % model_probabilities.sum())
    # renormalize?

    # convolve errors
    norm = model_probabilities.sum()
    model_probabilities = convolve_errors(model_probabilities, g_errors[0], r_errors[:, 0])
    model_probabilities *= (norm / model_probabilities.sum())  # keep same normalization
    # print('final model probs norm = %.8f' % model_probabilities.sum())

    if False:
        # checks - correct when full_probabilities not normed and model_probabilities not convolved with error: sum(model_probabilities) = expected # stars
        observable_fraction = calculate_observable_fraction(iso, mass_min)
        full_probabilities *= BIN_SIZE**2
        print()
        print('bin model')
        print(model_probabilities.sum() * TRUE_RICHNESS)
        print(full_probabilities.sum() * TRUE_RICHNESS)
        print(observable_fraction)
        print(model_probabilities.sum() / full_probabilities.sum())
        print()

    # model_probabilities[model_probabilities == 0] = model_probabilities[model_probabilities > 0].min() # BEST CHOICE HERE?
    return model_probabilities


###############################


def log_likelihood(model_parameters, binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors):
    global COUNTER
    COUNTER += 1.
    if not COUNTER % 1000:
        print(COUNTER)

    # model_parameters = [model_parameters[0], model_parameters[1], TRUE_AGE, TRUE_METALLICITY]
    if model_parameters[0] < 0 or model_parameters[1] < 10 or model_parameters[1] > 30 or model_parameters[2] < 0.1 or model_parameters[2] > 15 or model_parameters[3] < 1e-4 or model_parameters[3] > 0.02:
        # print("Out of range")
        return -np.inf

    global DISTANCE_MODULUS, AGE, METALLICITY, MODEL_PROBABILITIES
    if model_parameters[1] == DISTANCE_MODULUS and model_parameters[2] == AGE and model_parameters[3] == METALLICITY:
        # print("Not updating")
        model_probabilities = MODEL_PROBABILITIES
    else:
        # print("Updating")
        model_probabilities = bin_model(model_parameters, g_grid, r_grid, g_errors, r_errors)

        # update parameters
        DISTANCE_MODULUS = model_parameters[1]
        AGE = model_parameters[2]
        METALLICITY = model_parameters[3]
        MODEL_PROBABILITIES = model_probabilities

    richness = model_parameters[0]
    expected_counts = background_probabilities + richness * model_probabilities
    # print("Expected counts: ", background_probabilities.sum(), model_probabilities.sum(), model_probabilities.sum()*richness, expected_counts.sum(), binned_data.sum())

    loglike = poisson.logpmf(binned_data, expected_counts)
    # loglike[loglike == -np.inf] = -1e30 # for now, check how much this number matters

    idx = (expected_counts == 0) & (binned_data != 0)
    if idx.sum() > 0:
        print('loglike test: ', idx.sum())
    # THIS SHOULD NOT BE NECESSARY: loglike[(expected_counts == 0) & (binned_data != 0)] = poisson.logpmf(0, 1)

    gr_grid = g_grid - r_grid
    mask = (gr_grid > -0.5) & (gr_grid < 1)

    # return loglike # TEST
    return np.sum(loglike[mask]), loglike


################################################################

def plot_pretty(dpi=175, fontsize=20, label_size=20):
    # import pyplot and set some parameters to make plots prettier

    plt.rc('savefig', dpi=dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', size=fontsize)
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)

    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    mpl.rcParams.update({'figure.autolayout': True})
    return


def plot_likelihood(loglike, richness, distance_modulus, age, metallicity, max_vals, save=True):
    r_max, m_max, a_max, z_max = max_vals
    r_idx = np.where(richness == r_max)
    m_idx = np.where(distance_modulus == m_max)
    a_idx = np.where(age == a_max)
    z_idx = np.where(metallicity == z_max)

    r_array = loglike[:, m_idx, a_idx, z_idx].flatten()
    m_array = loglike[r_idx, :, a_idx, z_idx].flatten()
    a_array = loglike[r_idx, m_idx, :, z_idx].flatten()
    z_array = loglike[r_idx, m_idx, a_idx, :].flatten()

    plt.figure()
    plt.plot(richness, r_array, c='seagreen', lw=3)
    plt.grid()
    plt.xlabel(r'$\mathrm{Richness}$')
    plt.ylabel(r'$\log L$')
    plt.xscale('log')
    if save:
        plt.savefig('../plots/likelihood/%s/richness.png' % STREAM)

    plt.figure()
    plt.plot(distance_modulus, m_array, c='seagreen', lw=3)
    plt.axvline(TRUE_DISTANCE_MODULUS, c='k', ls='--')
    plt.axhline(m_array[np.argmin(np.abs(distance_modulus - TRUE_DISTANCE_MODULUS))], c='k', ls='--')
    plt.grid()
    plt.xlabel(r'$\mathrm{Distance\ Modulus}$')
    plt.ylabel(r'$\log L$')
    if save:
        plt.savefig('../plots/likelihood/%s/distance_modulus.png' % STREAM)

    plt.figure()
    plt.plot(age, a_array, c='seagreen', lw=3)
    plt.axvline(TRUE_AGE, c='k', ls='--')
    plt.axhline(a_array[np.argmin(np.abs(age - TRUE_AGE))], c='k', ls='--')
    plt.grid()
    plt.xlabel(r'$\mathrm{Age\ (Gyr)}$')
    plt.ylabel(r'$\log L$')
    if save:
        plt.savefig('../plots/likelihood/%s/age.png' % STREAM)

    plt.figure()
    plt.plot(metallicity, z_array, c='seagreen', lw=3)
    plt.axvline(TRUE_METALLICITY, c='k', ls='--')
    plt.axhline(z_array[np.argmin(np.abs(metallicity - TRUE_METALLICITY))], c='k', ls='--')
    plt.grid()
    plt.xlabel(r'$\mathrm{Metallicity\ (Z)}$')
    plt.ylabel(r'$\log L$')
    if save:
        plt.savefig('../plots/likelihood/%s/metallicity.png' % STREAM)

    pass


def make_plots(binned_data, background_probabilities, model_probabilities, richness, loglike, g_grid, r_grid, true_richness):
    plt.ion()

    # data
    plt.figure()
    plt.pcolormesh(r_grid, g_grid, binned_data - background_probabilities)

    plt.colorbar()
    plt.xlabel('r')
    plt.ylabel('g')
    plt.title('data - background')

    # signal
    plt.figure()
    plt.pcolormesh(r_grid, g_grid, true_richness * model_probabilities)

    plt.colorbar()
    plt.xlabel('r')
    plt.ylabel('g')
    plt.title('signal')

    # likelihood
    plt.figure()
    plt.plot(richness, loglike - np.max(loglike))

    plt.axvline(true_richness, c='gray', alpha=0.5, label='True Value')
    plt.axvline(richness[np.where(loglike == np.max(loglike))[0]], ls='--', c='gray', alpha=0.5, label='Maximum Likelihood')

    plt.axhline(loglike[np.where(richness == true_richness)[0]] - np.max(loglike), c='gray', alpha=0.5)
    plt.axhline(0, ls='--', c='gray', alpha=0.5)

    plt.legend(loc='lower left')
    plt.xlabel('richness')
    plt.ylabel('logL')

    plt.show()


def plot_cmd(g_grid, r_grid, binned_data, isochrone_params, save=True):
    gr_grid = g_grid - r_grid
    gr_centers = np.unique(gr_grid)
    g_centers = np.unique(g_grid)
    GR, G = np.meshgrid(gr_centers, g_centers)

    cmd = np.zeros_like(GR)
    for i in range(binned_data.shape[0]):
        for j in range(binned_data.shape[1]):
            x1 = np.where(G == g_grid[i][j])[0][0]
            x2 = np.where(GR == gr_grid[i][j])[1][0]
            cmd[x1][x2] = binned_data[i][j]

    plt.figure()
    plt.pcolormesh(GR, G, cmd, cmap='viridis')
    plt.colorbar()
    plt.xlim(-0.5, 1.5)
    plt.ylim(G_MAX, G_MIN)

    colors = ['orange', 'cyan', 'seagreen']
    for i, params in enumerate(isochrone_params):
        richness, distance_modulus, age, metallicity = params
        
        # iso = isochrone_factory('Padova', distance_modulus=distance_modulus, age=age, z=metallicity)
        iso = isochrone_factory('Padova', distance_modulus=distance_modulus, age=age, z=metallicity, dirname=ISO_DIR)
        # iso.mag_2 -= 0.08 # DELETE
        plt.scatter(iso.color, iso.mag_1 + iso.distance_modulus, facecolor=colors[i], edgecolor='none', s=8)

    if save:
        plt.savefig('../plots/likelihood/cmd.png')


def simulate_stream(g_bins, r_bins, g_errors, r_errors, richness=1.e6, distance_modulus=16.21, age=11.5, metallicity=0.0004, scatter=0.1):
    # iso = isochrone_factory('Padova', distance_modulus=distance_modulus, age=age, z=metallicity)
    iso = isochrone_factory('Padova', distance_modulus=distance_modulus, age=age, z=metallicity, dirname=ISO_DIR)

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
    test = 1
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


def grid_search(binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors, richness, distance_modulus, age, metallicity, printing=1, plotting=0):
    ll_max = -np.inf
    loglike = np.zeros((len(richness), len(distance_modulus), len(age), len(metallicity)))

    r_max = 0
    m_max = 0
    a_max = 0
    z_max = 0

    for j, m in enumerate(distance_modulus):
        print('m - M = %.2f' % m)
        for k, a in enumerate(age):
            for l, z in enumerate(metallicity):
                for i, r in enumerate(richness):

                    model_parameters = [r, m, a, z]
                    log_like, _ = log_likelihood(model_parameters, binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors)
                    loglike[i, j, k, l] = log_like

                    if log_like == -np.inf:
                        print('Problem!', log_like)

                    if log_like > ll_max:
                        r_max = r
                        m_max = m
                        a_max = a
                        z_max = z
                        ll_max = log_like

                        if printing > 1:
                            print()
                            print('new max!')
                            print('loglike = ', log_like)
                            print(r)
                            print(m)
                            print(a)
                            print(z)
                            print()

    if printing > 0:
        print()
        print('best fit parameters:')
        print('richness = %.2e' % r_max)
        print('distance modulus = %.2f' % m_max)
        print('age = %.2f' % a_max)
        print('metallicity = %.5f' % z_max)
        print()

        print('true parameters:')
        print('richness = %.2e' % TRUE_RICHNESS)
        print('distance modulus = %.2f' % TRUE_DISTANCE_MODULUS)
        print('age = %.2f' % TRUE_AGE)
        print('metallicity = %.5f' % TRUE_METALLICITY)
        print()

    if plotting:
        plot_pretty(fontsize=15, label_size=15)
        plot_likelihood(loglike, richness, distance_modulus, age, metallicity, [r_max, m_max, a_max, z_max], save=True)

    return r_max, m_max, a_max, z_max, loglike


def setup(stream):
    g_min = G_MIN
    g_max = G_MAX
    r_min = R_MIN
    r_max = R_MAX
    bin_size = BIN_SIZE

    g_bins, r_bins, g_grid, r_grid = define_bins(g_min, g_max, r_min, r_max, bin_size)

    data, background_data = load_data(stream)
    #data = load_data2(RA, DEC, radius=RADIUS)
    background_data = load_data2(RA, DEC + 1.5, radius=RADIUS * np.sqrt(N_BACKGROUND))
    print('check: ', data.shape, background_data.shape)

    g_errors, r_errors = propagate_errors(data, g_bins, r_bins)
    g_errors, r_errors = np.meshgrid(g_errors, r_errors)  # temporary

    binned_data = bin_data(data, g_bins, r_bins)
    background_probabilities = bin_background(background_data, g_bins, r_bins)
    print('check2: ', np.sum(binned_data), np.sum(background_probabilities))

    return binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors


def setup_test(stream, scatter=2 * BIN_SIZE):
    g_min = G_MIN
    g_max = G_MAX
    r_min = R_MIN
    r_max = R_MAX
    bin_size = BIN_SIZE

    g_bins, r_bins, g_grid, r_grid = define_bins(g_min, g_max, r_min, r_max, bin_size)

    data, background_data = load_data(stream)

    g_errors, r_errors = propagate_errors(data, g_bins, r_bins)
    g_errors = np.full_like(g_errors, 2 * BIN_SIZE)
    r_errors = np.copy(g_errors)

    new_data = simulate_stream(g_bins, r_bins, g_errors, r_errors, richness=TRUE_RICHNESS, distance_modulus=TRUE_DISTANCE_MODULUS, age=TRUE_AGE, metallicity=TRUE_METALLICITY, scatter=scatter)
    binned_stream = bin_data(new_data, g_bins, r_bins)

    g_errors, r_errors = np.meshgrid(g_errors, r_errors)  # temporary

    background_probabilities = bin_background(background_data, g_bins, r_bins)
    # background_probabilities = np.zeros_like(binned_stream)

    # simulated_background = simulate_background(background_probabilities)
    # binned_data = simulated_background + binned_stream # simulated background based on actual background regions
    binned_data = bin_data(data, g_bins, r_bins) + binned_stream  # add simulated stream to strengthen real stream signal

    return binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors


################################################################

if __name__ == "__main__":

    # binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors = setup()
    # binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors = setup_test()

    ###############################

    grid = 1
    if grid:
        print('Grid search: %s' % STREAM)
        print('________________')
        richness = 10**np.arange(0, 6 + 0.5, 0.5)
        # richness = np.logspace(4, 5, 20)
        distance_modulus = np.arange(14.5, 19., 0.5)
        age = np.arange(10., 14., 0.5)
        metallicity = np.arange(0.0001, 0.001, 0.0002)

        binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors = setup(STREAM)
        r_max, m_max, a_max, z_max, loglike = grid_search(binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors,
                                                          richness, distance_modulus, age, metallicity, printing=1, plotting=0)

        # np.savetxt('TucIII_loglike.txt', loglike)

    '''
        rmax_array = []
        mmax_array = []
        amax_array = []
        zmax_array = []

        i_max = 100
        i = 0

        while i < i_max:
            print('i = %i' % i)
            binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors = setup_test(STREAM)
            r_max, m_max, a_max, z_max, loglike = grid_search(binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors, richness, distance_modulus, age, metallicity, printing=1, plotting=0)
            rmax_array.append(r_max)
            mmax_array.append(m_max)
            amax_array.append(a_max)
            zmax_array.append(z_max)
            print()
            i += 1
            if a_max == 10 or a_max == 13.5:
                break

        # np.savetxt('richness_new.txt', rmax_array)
        # np.savetxt('distance_modulus_new.txt', mmax_array)
        # np.savetxt('age_new.txt', amax_array)
        # np.savetxt('metallicity_new.txt', zmax_array)
    '''
    ###############################

    mcmc = 0
    if mcmc:
        print('MCMC: %s' % STREAM)
        print('________________')
        # model_parameters = [richness, distance_modulus, age, metallicity]
        ndim, nwalkers = 4, int(8)
        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        p0[:, 0] = p0[:, 0] * 100000  # richness: (0, 100000) # might want to log space?
        p0[:, 1] = p0[:, 1] * 10 + 12  # distance_modulus: (12, 22)
        p0[:, 2] = p0[:, 2] * 5 + 9  # age: (9, 14)
        p0[:, 3] = (p0[:, 3] * 9 + 1) * 1e-4  # metallicity: (0.0001, 0.001) # should metallicity only be allowed to use isochrone file values? (steps of 0.0001)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=[binned_data, background_probabilities, g_grid, r_grid, g_errors, r_errors], threads=4)

        # burn-in
        pos, prob, state = sampler.run_mcmc(p0, 100)
        sampler.reset()

        pos, prob, state = sampler.run_mcmc(pos, 1000)

        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

        if True:
            for i in range(ndim):
                plt.figure()
                plt.hist(sampler.flatchain[:, i], 100, color="k", histtype="step")
                plt.title("Dimension {0:d}".format(i))
                plt.savefig('likelihood_sgr_%i.png' % i)

            samples = sampler.chain.reshape(-1, ndim)
            import corner
            fig = corner.corner(samples, labels=["Richness", "m - M", "Age", "Z"], truths=[TRUE_RICHNESS, TRUE_DISTANCE_MODULUS, TRUE_AGE, TRUE_METALLICITY])
            # fig = corner.corner(samples, labels=["Richness", "m - M"], truths=[TRUE_RICHNESS, TRUE_DISTANCE_MODULUS])

            plt.savefig('likelihood_sgr_corner.png')
            plt.show()
