from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
# sys.path.append('/home/norashipp/projects/stream_fitting_1/utils')
# sys.path.append('/Users/nora/projects/stream_fitting_1/utils')
sys.path.append('/home/s1/nshipp/projects/stream_fitting/utils')

import astropy.units as u
from astropy.coordinates import SkyCoord

import scipy.optimize as sopt

import scipy.interpolate as sinterp

import astropy.io.fits as fitsio
import scipy

from rotation_matrix import phi12_rotmat, pmphi12, pmphi12_reflex

from stream_data import stream_phi12_pms, stream_dists, stream_lengths, stream_matrices, stream_widths, stream_vr_widths, stream_masses

import gal_uvw

try:
    STREAM = sys.argv[1].replace('_', ' ')
    VERSION = int(sys.argv[2])
    PHI1 = sys.argv[3]
except:
    STREAM = 'ATLAS'
    VERSION = 111
    PHI1 = 'mid'
if STREAM == '-f':
    STREAM = 'ATLAS'
    VERSION = 111
    PHI1 = 'mid'

PLOTTING = False

# OUTPUTDIR = '/project2/kicp/norashipp/projects/stream_fitting_1/output/galpot/galpot_stream_mcmc_new/'
OUTPUTDIR = '/data/des40.b/data/nshipp/stream_fitting/galpot/galpot_stream_lmc/output/'
# OUTPUTDIR = './'

JHELUM_POP = 1

print('Fitting %s...' % STREAM)


R_phi12_radec = np.asarray(stream_matrices[STREAM])
M_prog = stream_masses[STREAM]  # 2.e-6 # 1.e-3
sigma_prog = stream_masses[STREAM]

# M_NFW = 80.
# q_NFW = 1.
# rs_NFW = 16.
# R0 = 8.3
G = 43007.105731706317

if PHI1 == 'mid':
    phi1_prog = 0.
elif PHI1 == 'neg':
    phi1_prog = -stream_lengths[STREAM] / 2. - 2.
elif PHI1 == 'pos':
    phi1_prog = stream_lengths[STREAM] / 2. + 2.
else:
    print('Invalid PHI1.')
    phi1_prog = 0.

a_g = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                [-0.8734370902, -0.4448296300, -0.1980763734],
                [-0.4838350155, 0.7469822445, +0.4559837762]])


l_lmc = -79.5342631651 * np.pi / 180.
b_lmc = -32.8903260974 * np.pi / 180.

####################
# GALPOT POTENTIAL #
####################
mcmillan = np.genfromtxt('ProductionRunBig2_Nora_10.tab')
mcm = mcmillan[9]
# mcm = mcmillan[0]

fileout = open('pot/PJM17_{0}.Tpot'.format(0), 'w')

print(4, file=fileout)
print('{0:.5e} {1:.5f} {2:.5f} {3} {4}'.format(mcm[0], mcm[1], mcm[2], mcm[3], mcm[4]), file=fileout)
print('{0:.5e} {1:.5f} {2:.5f} {3} {4}'.format(mcm[5], mcm[6], mcm[7], mcm[8], mcm[9]), file=fileout)
print('5.31319e+07 7 -0.085 4 0', file=fileout)
print('2.17995e+09 1.5 -0.045 12 0', file=fileout)

print(2, file=fileout)
print('{0:.5e} {1:.5f} {2:.5f} {3} {4} {5}'.format(mcm[20], mcm[21], mcm[22], mcm[23], mcm[24], mcm[25]), file=fileout)
print('{0:.5e} {1:.5f} {2:.5f} {3} {4} {5}'.format(mcm[26], mcm[27], mcm[28], mcm[29], mcm[30], mcm[31]), file=fileout)

Usun = mcm[32] * 1000.
Vsun = mcm[33] * 1000.
Wsun = mcm[34] * 1000.
R0 = mcm[-6]
V0 = mcm[-5]

M200 = 4. * np.pi * mcm[26] * mcm[30]**3. * (np.log(1. + mcm[-9] / mcm[30]) - mcm[-9] / (mcm[-9] + mcm[30])) / (1.e10)

c200 = mcm[-9] / mcm[30]
rs = mcm[30]

M_NFW = M200
rs_NFW = rs
c_NFW = c200

fileout.close()


def chi2_eval(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog, phi2_prog, M_LMC, sigma_extra):
    if M_LMC > 2.:
        rs_LMC = np.sqrt(G * M_LMC * 8.7 / 91.7**2.) - 8.7
    else:
        rs_LMC = np.sqrt(G * 2. * 8.7 / 91.7**2.) - 8.7

    mu_alpha_lmc = np.random.normal(1.91, 0. * 0.02)
    mu_delta_lmc = np.random.normal(0.229, 0. * 0.047)
    rv_lmc = np.random.normal(262.2, 0. * 3.4)
    dist_lmc = np.random.normal(49970., 0. * 1126.)

    tmax = 5.

    pid = os.getpid()

    lhood = chi2_worker(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog, phi2_prog, sigma_extra,
                        mu_alpha_lmc, mu_delta_lmc, rv_lmc, dist_lmc, M_LMC, rs_LMC, tmax, pid)
    return lhood


def chi2_worker(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog, phi2_prog, sigma_extra, mu_alpha_lmc, mu_delta_lmc, rv_lmc, dist_lmc, M_LMC, rs_LMC, tmax, pid):

    vec_phi12_prog = np.array([np.cos(phi1_prog * np.pi / 180.) * np.cos(phi2_prog * np.pi / 180.), np.sin(
        phi1_prog * np.pi / 180.) * np.cos(phi2_prog * np.pi / 180.), np.sin(phi2_prog * np.pi / 180.)])

    vec_radec_prog = np.linalg.solve(R_phi12_radec, vec_phi12_prog)

    ra_prog = np.arctan2(vec_radec_prog[1], vec_radec_prog[0]) * 180. / np.pi
    dec_prog = np.arcsin(
        vec_radec_prog[2] / np.linalg.norm(vec_radec_prog)) * 180. / np.pi

    gc_prog = SkyCoord(ra=ra_prog * u.degree,
                       dec=dec_prog * u.degree, frame='icrs')

    l_prog = np.array(gc_prog.galactic.l)
    b_prog = np.array(gc_prog.galactic.b)

    # vlsr = np.array([11.1, 245, 7.3])
    vlsr = np.array([Usun, V0 + Vsun, Wsun])

    M_UVW_muphi12_prog = np.array([[np.cos(phi1_prog * np.pi / 180.) * np.cos(phi2_prog * np.pi / 180.), -np.sin(phi1_prog * np.pi / 180.), -np.cos(phi1_prog * np.pi / 180.) * np.sin(phi2_prog * np.pi / 180.)], [np.sin(
        phi1_prog * np.pi / 180.) * np.cos(phi2_prog * np.pi / 180.), np.cos(phi1_prog * np.pi / 180.), -np.sin(phi1_prog * np.pi / 180.) * np.sin(phi2_prog * np.pi / 180.)], [np.sin(phi2_prog * np.pi / 180.), 0., np.cos(phi2_prog * np.pi / 180.)]])

    k_mu = 4.74047

    uvw_stationary = -vlsr

    vec_vr_muphi1_muphi2_stationary = np.dot(
        M_UVW_muphi12_prog.T, np.dot(R_phi12_radec, np.dot(a_g, uvw_stationary)))
    vec_vr_muphi1_muphi2_stationary[
        0] = 0.  # no correction for radial velocity, i want our radial velocity to be the los one

    # + vec_vr_muphi1_muphi2_stationary
    vec_vr_muphi1_muphi2_prog = np.array(
        [rv_prog, k_mu * dist_prog * mu_phi1cosphi2_prog, k_mu * dist_prog * mu_phi2_prog])

    vx_prog, vy_prog, vz_prog = np.dot(a_g.T, np.dot(
        R_phi12_radec.T, np.dot(M_UVW_muphi12_prog, vec_vr_muphi1_muphi2_prog))) + vlsr

    x_prog, y_prog, z_prog = np.array([-R0, 0., 0.]) + dist_prog * np.array([np.cos(l_prog * np.pi / 180.) * np.cos(
        b_prog * np.pi / 180.), np.sin(l_prog * np.pi / 180.) * np.cos(b_prog * np.pi / 180.), np.sin(b_prog * np.pi / 180.)])

    vx, vy, vz = vx_prog, vy_prog, vz_prog

    x, y, z = x_prog, y_prog, z_prog

    ###########

    gc = SkyCoord(b=b_lmc * u.radian, l=l_lmc * u.radian, frame='galactic')

    x_lmc, y_lmc, z_lmc = np.array([-R0, 0., 0.]) + dist_lmc / 1000. * np.array(
        [np.cos(l_lmc) * np.cos(b_lmc), np.sin(l_lmc) * np.cos(b_lmc), np.sin(b_lmc)])

    vtan = 0.
    vx_lmc, vy_lmc, vz_lmc = gal_uvw.gal_uvw(distance=dist_lmc, ra=np.array(gc.icrs.ra), dec=np.array(
        gc.icrs.dec), lsr=np.array([-Usun, vtan, Wsun]), pmra=mu_alpha_lmc, pmdec=mu_delta_lmc, vrad=rv_lmc)

    vy_lmc += V0 + Vsun
    vx_lmc = -vx_lmc

    ###########

    # os.system('./a.out {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(x, y, z, vx, vy, vz, M_NFW, M_LMC, rs_LMC, pid))
    os.system('./a.out {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20}'.format(
        x, y, z, vx, vy, vz, M_LMC, rs_LMC, x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc, vz_lmc, M_prog, sigma_prog, tmax, M_NFW, rs_NFW, c_NFW, pid))
    data = np.genfromtxt(OUTPUTDIR + 'final_stream_{0}.txt'.format(pid))

    if len(data) == 0:
        return -np.inf

    pos = data[:, :3]
    vel = data[:, 3:6]

    pos = pos - np.array([-R0, 0., 0.])

    theta = np.arctan2(pos[:, 1], pos[:, 0])
    theta = np.mod(theta, 2. * np.pi)
    theta[theta > np.pi] -= 2. * np.pi
    phi = np.arcsin(pos[:, 2] / (pos[:, 0]**2. +
                                 pos[:, 1]**2. + pos[:, 2]**2.)**0.5)

    l = 180. / np.pi * theta
    b = 180. / np.pi * phi

    gc = SkyCoord(l=l * u.degree, b=b * u.degree, frame='galactic')

    vec_radec = np.array([np.cos(gc.icrs.ra) * np.cos(gc.icrs.dec),
                          np.sin(gc.icrs.ra) * np.cos(gc.icrs.dec), np.sin(gc.icrs.dec)])

    vec_phi12 = np.dot(R_phi12_radec, vec_radec).T

    phi1 = np.arctan2(vec_phi12[:, 1], vec_phi12[:, 0]) * 180. / np.pi
    phi2 = np.arcsin(vec_phi12[:, 2]) * 180. / np.pi

    vel -= vlsr

    alpha = np.array(gc.icrs.ra)  # * np.pi / 180.
    delta = np.array(gc.icrs.dec)  # * np.pi / 180.

    r_stream = np.sum(pos[:, axis]**2. for axis in range(3))**0.5

    R_phi12_a_g = np.dot(R_phi12_radec, a_g)

    vr_stream = np.sum((np.cos(phi1 * np.pi / 180.) * np.cos(phi2 * np.pi / 180.) * R_phi12_a_g[0, axis] + np.sin(phi1 * np.pi / 180.) * np.cos(
        phi2 * np.pi / 180.) * R_phi12_a_g[1, axis] + np.sin(phi2 * np.pi / 180.) * R_phi12_a_g[2, axis]) * vel[:, axis] for axis in range(3))

    mu_phi1_cos_phi2_stream = 1. / (k_mu * r_stream) * np.sum((-np.sin(phi1 * np.pi / 180.) * R_phi12_a_g[
        0, axis] + np.cos(phi1 * np.pi / 180.) * R_phi12_a_g[1, axis]) * vel[:, axis] for axis in range(3))

    mu_phi2_stream = 1. / (k_mu * r_stream) * np.sum((-np.cos(phi1 * np.pi / 180.) * np.sin(phi2 * np.pi / 180.) * R_phi12_a_g[0, axis] - np.sin(
        phi1 * np.pi / 180.) * np.sin(phi2 * np.pi / 180.) * R_phi12_a_g[1, axis] + np.cos(phi2 * np.pi / 180.) * R_phi12_a_g[2, axis]) * vel[:, axis] for axis in range(3))

    nvlsr = -vlsr

    mu_phi1_cos_phi2_stream_corr = mu_phi1_cos_phi2_stream - 1. / (k_mu * r_stream) * np.sum((-np.sin(phi1 * np.pi / 180.) * R_phi12_a_g[
        0, axis] + np.cos(phi1 * np.pi / 180.) * R_phi12_a_g[1, axis]) * nvlsr[axis] for axis in range(3))

    mu_phi2_stream_corr = mu_phi2_stream - 1. / (k_mu * r_stream) * np.sum((-np.cos(phi1 * np.pi / 180.) * np.sin(phi2 * np.pi / 180.) * R_phi12_a_g[0, axis] - np.sin(
        phi1 * np.pi / 180.) * np.sin(phi2 * np.pi / 180.) * R_phi12_a_g[1, axis] + np.cos(phi2 * np.pi / 180.) * R_phi12_a_g[2, axis]) * nvlsr[axis] for axis in range(3))

    #############
    # LOAD DATA #
    #############

    hdul = fitsio.open('track_atlas_06112019.fits')
    data_track = hdul[1].data

    hdul = fitsio.open('atlas_mem_v3.fits')
    data_S5 = hdul[1].data

    # if STREAM == 'Jhelum':
    #     s5_data = np.loadtxt('../%s_pop%i_S5.txt' %
    #                          (STREAM.replace(' ', '_'), JHELUM_POP))
    # else:
    #     s5_data = np.loadtxt('../%s_S5.txt' % STREAM.replace(' ', '_'))

    # phi1_s5, phi2_s5 = phi12_rotmat(
    #     s5_data[:, 0], s5_data[:, 1], R_phi12_radec)

    phi1_s5, phi2_s5 = phi12_rotmat(data_S5['ra'], data_S5['dec'], R_phi12_radec)

    # LIKELIHOOD
    chi2 = 0.
    dof = 0.

    dL0 = 1.

    #########
    # Track #
    #########

    chi2_st = 0.

    phi1_track = data_track['fi1']
    phi2_track = data_track['fi2']
    phi2_track_std = data_track['fi2_std']

    for counter in range(len(phi1_track)):
        t_var = phi1_track[counter]

        mask = np.where((phi1 > t_var - dL0 / 2.) * (phi1 < t_var + dL0 / 2.))[0]
        if len(mask) > 5:
            phi1_data = phi1[mask]
            phi2_data = phi2[mask]
            phi1_mid = t_var

            def lhood(args):
                a_guess = args[0]
                b_guess = args[1]
                sigma_guess = args[2]
                sigma_eff = (sigma_guess**2.)**0.5
                phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess
                return np.sum(np.log(sigma_eff) + (phi2_data - phi2_guess)**2. / (2. * sigma_eff**2.))

            def dlhood(args):
                a_guess = args[0]
                b_guess = args[1]
                sigma_guess = args[2]
                sigma_eff = (sigma_guess**2.)**0.5
                phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess
                dlogLda = -np.sum((phi2_data - phi2_guess) / (sigma_eff**2.) * (phi1_data - phi1_mid))
                dlogLdb = -np.sum((phi2_data - phi2_guess) / (sigma_eff**2.))
                dlogLdsigma = np.sum(1. / sigma_guess - (phi2_data - phi2_guess)**2. / (sigma_guess**3.))
                return np.array([dlogLda, dlogLdb, dlogLdsigma])

            result = sopt.fmin_bfgs(lhood, [0., np.mean(phi2_data), np.std(phi2_data)], fprime=dlhood, full_output=True, disp=False)

            a_estimate = result[0][0]
            a_error = scipy.linalg.sqrtm(result[3])[0][0]

            b_estimate = result[0][1]
            b_error = scipy.linalg.sqrtm(result[3])[1][1]

            sigma_estimate = result[0][2]
            sigma_error = scipy.linalg.sqrtm(result[3])[2][2]

            mu_data = b_estimate
            sigma_data = sigma_estimate  # because comparing to individual data points, not binned fit

        # elif len(mask) == 1:
        #     continue
        else:
            chi2_st += 100
            continue

        # COMPARE SIMULATION TO DATA
        mu_data = b_estimate
        sigma_data = sigma_estimate
        sigma_mu = b_error

        mu_obs = phi2_track[counter]
        sigma_obs = phi2_track_std[counter]
        sigma = (sigma_mu**2. + sigma_obs**2.)**0.5
        chi2_st += np.log(np.sqrt(2. * np.pi * sigma**2.)) + 0.5 * (mu_data - mu_obs)**2. / (sigma)**2.

    chi2 += chi2_st

    ######
    # PM #
    ######

    pmra, pmdec = pmphi12(phi1, phi2, mu_phi1_cos_phi2_stream,
                          mu_phi2_stream, R_phi12_radec.T)

    chi2_pmra = 0.

    for counter in range(len(phi1_s5)):
        t_var = phi1_s5[counter]
        mask = np.where((phi1 > t_var - 0.5) * (phi1 < t_var + 0.5))[0]  # approx length fit with gmm
        if len(mask) > 5:

            # COMPUTE MAX LIKELIHOOD LOCATION OF STREAM OBSERVABLE
            phi1_data = phi1[mask]
            phi2_data = pmra[mask]  # This needs to change for each observable  #########
            phi1_mid = t_var

            def lhood(args):
                a_guess = args[0]
                b_guess = args[1]
                sigma_guess = args[2]
                sigma_eff = (sigma_guess**2.)**0.5
                phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess
                return np.sum(np.log(sigma_eff) + (phi2_data - phi2_guess)**2. / (2. * sigma_eff**2.))

            def dlhood(args):
                a_guess = args[0]
                b_guess = args[1]
                sigma_guess = args[2]
                sigma_eff = (sigma_guess**2.)**0.5
                phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess
                dlogLda = -np.sum((phi2_data - phi2_guess) / (sigma_eff**2.) * (phi1_data - phi1_mid))
                dlogLdb = -np.sum((phi2_data - phi2_guess) / (sigma_eff**2.))
                dlogLdsigma = np.sum(1. / sigma_guess - (phi2_data - phi2_guess)**2. / (sigma_guess**3.))
                return np.array([dlogLda, dlogLdb, dlogLdsigma])

            result = sopt.fmin_bfgs(lhood, [0., np.mean(phi2_data), np.std(phi2_data)], fprime=dlhood, full_output=True, disp=False)

            a_estimate = result[0][0]
            a_error = scipy.linalg.sqrtm(result[3])[0][0]

            b_estimate = result[0][1]
            b_error = scipy.linalg.sqrtm(result[3])[1][1]

            sigma_estimate = result[0][2]
            sigma_error = scipy.linalg.sqrtm(result[3])[2][2]

            mu_data = b_estimate
            sigma_data = sigma_estimate  # because comparing to individual data points, not binned fit

        # elif len(mask) == 1:
        #     continue
        else:
            chi2_pmra += 100
            continue

        # COMPARE SIMULATION TO DATA
        mu_obs = data_S5['pmra'][counter]
        sigma_obs = data_S5['pmra_error'][counter]
        sigma = (sigma_data**2. + sigma_obs**2.)**0.5
        chi2_pmra += np.log(np.sqrt(2. * np.pi * sigma**2.)) + 0.5 * (mu_data - mu_obs)**2. / (sigma)**2.

    chi2 += chi2_pmra

    #################

    chi2_pmdec = 0.

    for counter in range(len(phi1_s5)):
        t_var = phi1_s5[counter]
        mask = np.where((phi1 > t_var - 0.5) * (phi1 < t_var + 0.5))[0]  # approx length fit with gmm
        if len(mask) > 5:

            # COMPUTE MAX LIKELIHOOD LOCATION OF STREAM OBSERVABLE
            phi1_data = phi1[mask]
            phi2_data = pmdec[mask]  # This needs to change for each observable  #########
            phi1_mid = t_var

            def lhood(args):
                a_guess = args[0]
                b_guess = args[1]
                sigma_guess = args[2]
                sigma_eff = (sigma_guess**2.)**0.5
                phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess
                return np.sum(np.log(sigma_eff) + (phi2_data - phi2_guess)**2. / (2. * sigma_eff**2.))

            def dlhood(args):
                a_guess = args[0]
                b_guess = args[1]
                sigma_guess = args[2]
                sigma_eff = (sigma_guess**2.)**0.5
                phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess
                dlogLda = -np.sum((phi2_data - phi2_guess) / (sigma_eff**2.) * (phi1_data - phi1_mid))
                dlogLdb = -np.sum((phi2_data - phi2_guess) / (sigma_eff**2.))
                dlogLdsigma = np.sum(1. / sigma_guess - (phi2_data - phi2_guess)**2. / (sigma_guess**3.))
                return np.array([dlogLda, dlogLdb, dlogLdsigma])

            result = sopt.fmin_bfgs(lhood, [0., np.mean(phi2_data), np.std(phi2_data)], fprime=dlhood, full_output=True, disp=False)

            a_estimate = result[0][0]
            a_error = scipy.linalg.sqrtm(result[3])[0][0]

            b_estimate = result[0][1]
            b_error = scipy.linalg.sqrtm(result[3])[1][1]

            sigma_estimate = result[0][2]
            sigma_error = scipy.linalg.sqrtm(result[3])[2][2]

            mu_data = b_estimate
            sigma_data = sigma_estimate  # because comparing to individual data points, not binned fit

        # elif len(mask) == 1:
        #     continue
        else:
            chi2_pmdec += 100
            continue

        # COMPARE SIMULATION TO DATA
        mu_obs = data_S5['pmdec'][counter]
        sigma_obs = data_S5['pmdec_error'][counter]
        sigma = (sigma_data**2. + sigma_obs**2.)**0.5
        chi2_pmdec += np.log(np.sqrt(2. * np.pi * sigma**2.)) + 0.5 * (mu_data - mu_obs)**2. / (sigma)**2.

    chi2 += chi2_pmdec

    #########################
    ### Radial Velocities ###
    #########################

    chi2_vr = 0.

    if PLOTTING:
        vr_mu_data = []
        vr_mu_obs = []
        vr_sig_obs = []
        phi1_mu_obs = []

    for counter in range(len(phi1_s5)):
        t_var = phi1_s5[counter]
        mask = np.where((phi1 > t_var - 0.5) * (phi1 < t_var + 0.5))[0]
        if len(mask) > 5:

            # COMPUTE MAX LIKELIHOOD LOCATION OF STREAM OBSERVABLE
            phi1_data = phi1[mask]
            phi2_data = vr_stream[mask]
            phi1_mid = t_var

            def lhood(args):
                a_guess = args[0]
                b_guess = args[1]
                sigma_guess = args[2]
                sigma_eff = (sigma_guess**2.)**0.5
                phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess
                return np.sum(np.log(sigma_eff) + (phi2_data - phi2_guess)**2. / (2. * sigma_eff**2.))

            def dlhood(args):
                a_guess = args[0]
                b_guess = args[1]
                sigma_guess = args[2]
                sigma_eff = (sigma_guess**2.)**0.5
                phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess
                dlogLda = -np.sum((phi2_data - phi2_guess) / (sigma_eff**2.) * (phi1_data - phi1_mid))
                dlogLdb = -np.sum((phi2_data - phi2_guess) / (sigma_eff**2.))
                dlogLdsigma = np.sum(1. / sigma_guess - (phi2_data - phi2_guess)**2. / (sigma_guess**3.))
                return np.array([dlogLda, dlogLdb, dlogLdsigma])

            result = sopt.fmin_bfgs(lhood, [0., np.mean(phi2_data), np.std(phi2_data)], fprime=dlhood, full_output=True, disp=False)

            a_estimate = result[0][0]
            a_error = scipy.linalg.sqrtm(result[3])[0][0]

            b_estimate = result[0][1]
            b_error = scipy.linalg.sqrtm(result[3])[1][1]

            sigma_estimate = result[0][2]
            sigma_error = scipy.linalg.sqrtm(result[3])[2][2]

            mu_data = b_estimate
            sigma_data = sigma_estimate  # because comparing to individual data points, not binned fit

            if PLOTTING:
                vr_mu_obs.append(vr_s5[:, 0][counter])
                vr_sig_obs.append(vr_s5[:, 1][counter])
                vr_mu_data.append(mu_data)
                phi1_mu_obs.append(t_var)

        # elif len(mask) == 1:
        #     continue
        else:
            chi2_vr += 100
            continue

        # COMPARE SIMULATION TO DATA
        mu_obs = data_S5['vel_calib'][counter]
        sigma_obs = data_S5['vel_calib_std'][counter]
        sigma = (sigma_data**2. + sigma_obs**2.)**0.5
        chi2_vr += np.log(np.sqrt(2. * np.pi * sigma**2.)) + 0.5 * (mu_data - mu_obs)**2. / (sigma)**2.

    chi2 += chi2_vr

    if PLOTTING:
        vr_mu_data = np.asarray(vr_mu_data)
        vr_mu_obs = np.asarray(vr_mu_obs)
        vr_sig_obs = np.asarray(vr_sig_obs)
        phi1_mu_obs = np.asarray(phi1_mu_obs)

    ##########################################################################

    if STREAM == 'Jhelum':
        fileout = open(OUTPUTDIR + 'save_chi2_%s_%i_%i_%s_%i.txt' %
                       (STREAM.replace(' ', '_'), JHELUM_POP, VERSION, PHI1, pid), 'a')
    else:
        fileout = open(OUTPUTDIR + 'save_chi2_%s_%i_%s_%i.txt' %
                       (STREAM.replace(' ', '_'), VERSION, PHI1, pid), 'a')

    print(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog,
          phi2_prog, M_LMC, M_NFW, chi2_st, chi2_pmra, chi2_pmdec, chi2_vr, chi2, file=fileout)

    fileout.close()

    if PLOTTING:

        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(12, 12))
        fig.subplots_adjust(hspace=0)

        ax[0].set_title(r'$\mu_1 \cos \phi_2 = %.2f,\ \mu_2 = %.2f,\ v_{\mathrm{r}} = %.3f,\ ,\phi_2 = %.2f,\ M_{\rm LMC}=%.3e,\ \log L = %.3f$' % (
            mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, phi2_prog, M_LMC * 1e10, -chi2))

        ax[0].errorbar(phi1_s5, phi2_s5, yerr=np.sqrt(stream_widths[STREAM]**2 + sigma_extra**2) * np.ones(len(phi1_s5)),
                       c='r', marker='o', ecolor='r', markerfacecolor='r', capsize=3, linestyle='None', ms=1)
        ax[0].plot(phi1_prog, phi2_prog, 'g*', ms=20)

        xlim = ax[0].get_xlim()
        ax[0].set_xlim(xlim[0], xlim[1])

        ax[0].scatter(phi1, phi2, s=1, c='navy')

        ax[1].scatter(phi1, r_stream, s=1, c='navy')
        ax[1].axhline(stream_dists[STREAM], ls='--', c='r', lw=2)

        # ax[2].scatter(phi1, vr_stream, s=1, c='navy')
        # ax[2].errorbar(phi1_s5, vr_s5[:, 0], yerr=vr_s5[:, 1], c='r', marker='o',
        # ecolor='r', markerfacecolor='r', capsize=3, linestyle='None', ms=1)
        ax[2].axhline(0 + stream_vr_widths[STREAM], ls='--', lw=2, c='navy')
        ax[2].axhline(0 - stream_vr_widths[STREAM], ls='--', lw=2, c='navy')
        ax[2].set_ylim(-50, 50)
        ax[2].errorbar(phi1_mu_obs, vr_mu_obs - vr_mu_data, yerr=vr_sig_obs, c='r', marker='o',
                       ecolor='r', markerfacecolor='r', capsize=3, linestyle='None', ms=1)
        # ax[2].errorbar(phi1_s5, vr_s5[:, 0] - vr_mu_data, yerr=vr_s5[:, 1], c='r', marker='o',
        #                ecolor='r', markerfacecolor='r', capsize=3, linestyle='None', ms=1)

        '''ax[3].scatter(phi1,mu_phi1_cos_phi2_stream,s=1)
        ax[3].scatter(phi1,mu_phi2_stream,s=1)'''

        ax[3].scatter(phi1, pmra, s=1, c='navy')
        ax[4].scatter(phi1, pmdec, s=1, c='navy')

        ax[3].errorbar(phi1_s5, pmra_s5[:, 0], yerr=pmra_s5[:, 1], c='r', marker='o',
                       ecolor='r', markerfacecolor='r', capsize=3, linestyle='None', ms=1)

        ax[4].errorbar(phi1_s5, pmdec_s5[:, 0], yerr=pmdec_s5[:, 1], c='r', marker='o',
                       ecolor='r', markerfacecolor='r', capsize=3, linestyle='None', ms=1)

        ax[4].set_xlabel(r'$\phi_1\ (^\circ)$')
        ax[0].set_ylabel(r'$\phi_2\ (^\circ)$')
        ax[1].set_ylabel(r'$r\ \mathrm{(kpc)}$')
        ax[2].set_ylabel(r'$\Delta v_r\ \mathrm{(km/s)}$')
        # ax[2].set_ylabel(r'$v_r\ \mathrm{(km/s)}$')
        ax[3].set_ylabel(r'$\mu_\alpha^*\ \mathrm{(mas/yr)}$')
        ax[4].set_ylabel(r'$\mu_\delta\ \mathrm{(mas/yr)}$')

        # ax[3].set_ylim(int(np.mean(pmra[np.abs(phi1) < 10]) - 5),
        #                int(np.mean(pmra[np.abs(phi1) < 10]) + 5))
        # ax[4].set_ylim(int(np.mean(pmdec[np.abs(phi1) < 10]) - 5),
        #                int(np.mean(pmdec[np.abs(phi1) < 10]) + 5))

        # ax[3].set_ylim(int(np.mean(pmra[np.abs(phi1) < 10]) - 2),
        #                int(np.mean(pmra[np.abs(phi1) < 10]) + 2))
        # ax[4].set_ylim(int(np.mean(pmdec[np.abs(phi1) < 10]) - 2),
        #                int(np.mean(pmdec[np.abs(phi1) < 10]) + 2))

        # ax[0].set_xlim(-20., 20.)
        ax[0].set_ylim(-2., 2)

        plt.tight_layout()

        plt.savefig('../plots/%s_stream_%2f_%i.png' %
                    (STREAM.replace(' ', '_'), phi1_prog, pid), bbox_inches='tight')

    if chi2 == 0. or chi2_st == 0. or chi2_pmra == 0. or chi2_pmdec == 0. or chi2_vr == 0.:
        return -np.inf
    else:
        return -chi2  # , -chi2_st, -chi2_pmra, -chi2_pmdec, -chi2_vr
