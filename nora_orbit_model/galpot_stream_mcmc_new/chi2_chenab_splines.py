from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
# sys.path.append('/home/norashipp/projects/stream_fitting_1/utils')
# sys.path.append('/Users/nora/projects/stream_fitting_1/utils')
sys.path.insert(0, '/home/s1/nshipp/projects/stream_fitting/utils')

import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fitsio

import scipy
import scipy.optimize as sopt
import scipy.interpolate as sinterp
from scipy.misc import derivative
from scipy.stats import norm
from scipy.stats import truncnorm

from rotation_matrix import phi12_rotmat, pmphi12, pmphi12_reflex, Mrot

from stream_data import stream_phi12_pms, stream_dists, stream_lengths, stream_matrices, stream_widths, stream_vr_widths, stream_masses, stream_sigmas

import gal_uvw

try:
    STREAM = sys.argv[1].replace('_', ' ')
    VERSION = int(sys.argv[2])
    PHI1 = sys.argv[3]
except:
    STREAM = 'Chenab'
    VERSION = 111
    PHI1 = 'mid'
if STREAM == '-f':
    STREAM = 'Chenab'
    VERSION = 111
    PHI1 = 'mid'

PLOTTING = False
DIST = True
if DIST:
    print('Including distance in likelihood!')

# OUTPUTDIR = '/project2/kicp/norashipp/projects/stream_fitting_1/output/galpot/galpot_stream_mcmc_new/'
# OUTPUTDIR = '/data/des40.b/data/nshipp/stream_fitting/galpot/galpot_stream_lmc/output/'
OUTPUTDIR = '/data/des70.a/data/nshipp/stream_fitting/output/galpot_stream_mcmc_new/dec20/'
# OUTPUTDIR = './'

try:
    JHELUM_POP = int(sys.argv[4])
except:
    JHELUM_POP = 1

print('Fitting %s...' % STREAM)


# R_phi12_radec = np.asarray(stream_matrices[STREAM])
R_phi12_radec = Mrot(72, -14, 23.48788614)
M_prog = stream_masses[STREAM]  # 2.e-6 # 1.e-3
sigma_prog = stream_sigmas[STREAM]
print('M_prog = %.2e' % M_prog)


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
    try:
        phi1_prog = float(PHI1)
    except:
        print('Invalid PHI1.')
        phi1_prog = 0.
print('phi1 = %.2f' % phi1_prog)

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


def atlas_ridgeline(L):
    B = -0.00357383 * L ** 2 + 0.00593217 * L + 0.66463043
    return B


def dist2mod(distance):
    return 5. * (np.log10(np.array(distance) * 1.e3) - 1.)


def mod2dist(distance_modulus):
    return 10**(distance_modulus / 5. + 1.) / 1e3


def chi2_eval(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog, phi2_prog, M_LMC, sigma_track, sigma_vr, mu_alpha_lmc=1.91, mu_delta_lmc=0.229, rv_lmc=262.2, dist_lmc=49970., lmc_disk=True):
    if lmc_disk == True:
        a_MN_LMC = 1.5
        b_MN_LMC = 0.3
        M_MN_LMC = 0.3
        M_LMC = M_LMC - M_MN_LMC

        def F_MN_LMC(R):
            return G * M_MN_LMC / (R**2. + (a_MN_LMC + b_MN_LMC)**2.)**1.5 * R

        if M_LMC > 2.:
            rs_LMC = np.sqrt(G * M_LMC / (91.7**2. / 8.7 - F_MN_LMC(8.7))) - 8.7
        else:
            rs_LMC = np.sqrt(G * 2. * 8.7 / 91.7**2.) - 8.7

        print('LMC disk!')

    else:
        print('LMC no disk!')
        if M_LMC > 2.:
            rs_LMC = np.sqrt(G * M_LMC * 8.7 / 91.7**2.) - 8.7
        else:
            rs_LMC = np.sqrt(G * 2. * 8.7 / 91.7**2.) - 8.7

    # mu_alpha_lmc = np.random.normal(1.91, 0. * 0.02)
    # mu_delta_lmc = np.random.normal(0.229, 0. * 0.047)
    # rv_lmc = np.random.normal(262.2, 0. * 3.4)
    # dist_lmc = np.random.normal(49970., 0. * 1126.)

    tmax = 5.

    pid = os.getpid()

    lhood = chi2_worker(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog, phi2_prog, sigma_track, sigma_vr,
                        mu_alpha_lmc, mu_delta_lmc, rv_lmc, dist_lmc, M_LMC, rs_LMC, tmax, pid, lmc_disk)
    return lhood


def chi2_worker(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog, phi2_prog, sigma_track, sigma_vr, mu_alpha_lmc, mu_delta_lmc, rv_lmc, dist_lmc, M_LMC, rs_LMC, tmax, pid, lmc_disk):

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
    # print('./a.out {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20}'.format(
    #             x, y, z, vx, vy, vz, M_LMC, rs_LMC, x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc, vz_lmc, M_prog, sigma_prog, tmax, M_NFW, rs_NFW, c_NFW, pid))

    if lmc_disk:
        if STREAM in ['Tucana III', 'Tucana_III']:
            os.system('./a_tuc3.out {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20}'.format(
                x, y, z, vx, vy, vz, M_LMC, rs_LMC, x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc, vz_lmc, M_prog, sigma_prog, tmax, M_NFW, rs_NFW, c_NFW, pid))
        else:
            os.system('./a2000.out {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20}'.format(
                x, y, z, vx, vy, vz, M_LMC, rs_LMC, x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc, vz_lmc, M_prog, sigma_prog, tmax, M_NFW, rs_NFW, c_NFW, pid))
    else:
        if STREAM in ['Tucana III', 'Tucana_III']:
            os.system('./a_tuc3_nodisk.out {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20}'.format(
                x, y, z, vx, vy, vz, M_LMC, rs_LMC, x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc, vz_lmc, M_prog, sigma_prog, tmax, M_NFW, rs_NFW, c_NFW, pid))
        else:
            os.system('./a2000_nodisk.out {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20}'.format(
                x, y, z, vx, vy, vz, M_LMC, rs_LMC, x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc, vz_lmc, M_prog, sigma_prog, tmax, M_NFW, rs_NFW, c_NFW, pid))

    data = np.genfromtxt(OUTPUTDIR + 'final_stream_{0}.txt'.format(pid))

    if len(data) == 0:
        return -np.inf

    # PHASE CUT TO AVOID WRAPS
    idx = np.where(np.abs(data[:, 8]) < np.pi)
    data = data[idx]

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

    vel0 = np.copy(vel)
    vel -= vlsr

    alpha = np.array(gc.icrs.ra)  # * np.pi / 180.
    delta = np.array(gc.icrs.dec)  # * np.pi / 180.

    r_stream = np.sum(pos[:, axis]**2. for axis in range(3))**0.5

    R_phi12_a_g = np.dot(R_phi12_radec, a_g)

    vr_stream = np.sum((np.cos(phi1 * np.pi / 180.) * np.cos(phi2 * np.pi / 180.) * R_phi12_a_g[0, axis] + np.sin(phi1 * np.pi / 180.) * np.cos(
        phi2 * np.pi / 180.) * R_phi12_a_g[1, axis] + np.sin(phi2 * np.pi / 180.) * R_phi12_a_g[2, axis]) * vel[:, axis] for axis in range(3))

    # v_gsr
    vr_gsr_stream = np.sum((np.cos(phi1 * np.pi / 180.) * np.cos(phi2 * np.pi / 180.) * R_phi12_a_g[0, axis] + np.sin(phi1 * np.pi / 180.) * np.cos(
        phi2 * np.pi / 180.) * R_phi12_a_g[1, axis] + np.sin(phi2 * np.pi / 180.) * R_phi12_a_g[2, axis]) * vel0[:, axis] for axis in range(3))


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

    # LOAD SPLINES

    hdul = fitsio.open('../orphan_splines/orphan_track_bins.fits')
    track_arr = hdul[1].data

    hdul = fitsio.open('../orphan_splines/orphan_pm1_bins.fits')
    pm1_arr = hdul[1].data

    hdul = fitsio.open('../orphan_splines/orphan_pm2_bins.fits')
    pm2_arr = hdul[1].data

    hdul = fitsio.open('../orphan_splines/orphan_rv_bins.fits')
    rv_arr = hdul[1].data

    hdul = fitsio.open('../orphan_splines/orphan_dm.fits')
    dm_arr = hdul[1].data

    # Clip to Chenab Range
    # phi1_min, phi1_max = -51.05387166, -69.54987964
    phi1_min, phi1_max = -100, 101
    track_arr = track_arr[(track_arr['phi1'] > phi1_min) & (track_arr['phi1'] < phi1_max)]
    pm1_arr = pm1_arr[(pm1_arr['phi1'] > phi1_min) & (pm1_arr['phi1'] < phi1_max)]
    pm2_arr = pm2_arr[(pm2_arr['phi1'] > phi1_min) & (pm2_arr['phi1'] < phi1_max)]
    rv_arr = rv_arr[(rv_arr['phi1'] > phi1_min) & (rv_arr['phi1'] < phi1_max)]
    dm_arr = dm_arr[(dm_arr['phi1'] > phi1_min) & (dm_arr['phi1'] < phi1_max)]

    # if STREAM == 'Jhelum':
    #     s5_data = np.loadtxt('../%s_pop%i_S5.txt' %
    #                          (STREAM.replace(' ', '_'), JHELUM_POP))
    # else:
    #     s5_data = np.loadtxt('../%s_S5.txt' % STREAM.replace(' ', '_'))

    # #  CHENAB 36
    # # s5_data = np.loadtxt('../Chenab_S5_edr3.txt')
    # # s5_data = np.loadtxt('../Chenab_S5_cut.txt')

    # phi1_s5, phi2_s5 = phi12_rotmat(
    #     s5_data[:, 0], s5_data[:, 1], R_phi12_radec)

    # LIKELIHOOD
    chi2 = 0.
    dof = 0.
    dL0 = 2.

    #########
    # Track #
    #########

    chi2_st = 0.

    for counter in range(len(track_arr['phi1'])):
        t_var = track_arr['phi1'][counter]
        mask = np.where((phi1 > t_var - dL0 / 2.) * (phi1 < t_var + dL0 / 2.))[0]
        if len(mask) > 5:

            # COMPUTE MAX LIKELIHOOD LOCATION OF STREAM OBSERVABLE
            phi1_data = phi1[mask]
            phi2_data = phi2[mask]  # This needs to change for each observable  #########
            phi1_mid = t_var

            if STREAM == 'ATLAS':
                x1 = atlas_ridgeline(phi1_mid) - 1
                x2 = atlas_ridgeline(phi1_mid) + 1
            else:
                x1, x2 = -1, 1

            # CHENAB 39
            TRUNCATED = False

            if TRUNCATED:
                # Truncated Gaussian
                def lhood(args):
                    # print('x1, x2 = ', x1, x2)
                    a_guess = args[0]
                    b_guess = args[1]
                    sigma_guess = args[2]
                    sigma_eff = (sigma_guess**2.)**0.5
                    phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess
                    return np.sum(-np.log(np.sqrt(2 * np.pi) * truncnorm.pdf(phi2_data, x1, x2, phi2_guess, sigma_eff)))

                def dlhood(args):
                    a_guess = args[0]
                    b_guess = args[1]
                    sigma_guess = args[2]
                    sigma_eff = (sigma_guess**2.)**0.5
                    phi2_guess = a_guess * (phi1_data - phi1_mid) + b_guess

                    lhood_a = lambda a: lhood((a, b_guess, sigma_guess))
                    lhood_b = lambda b: lhood((a_guess, b, sigma_guess))
                    lhood_sigma = lambda sig: lhood((a_guess, b_guess, sig))

                    dlogLda = derivative(lhood_a, a_guess, dx=1e-6)
                    dlogLdb = derivative(lhood_b, b_guess, dx=1e-6)
                    dlogLdsigma = derivative(lhood_sigma, sigma_guess, dx=1e-6)
                    return np.array([dlogLda, dlogLdb, dlogLdsigma])
            else:
                # Original, Analytic
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
            sigma_data = b_error  # sigma_estimate  # because comparing to individual data points, not binned fit

        # elif len(mask) == 1:
        #     continue
        else:
            chi2_st += 100
            continue

        # COMPARE SIMULATION TO DATA
        mu_obs = track_arr['phi2'][counter]
        sigma_obs = track_arr['ephi2'][counter]
        sigma = (sigma_data**2. + sigma_obs**2.)**0.5  # + sigma_track**2.)**0.5 # Keep sigma track param?
        chi2_st += np.log(np.sqrt(2. * np.pi * sigma**2.)) + 0.5 * (mu_data - mu_obs)**2. / (sigma)**2.

        # print('widths: %.2f, %.2f, %.2f, %.2f' %(stream_widths[STREAM], sigma_data, sigma_track, sigma))

    chi2 += chi2_st

    ######
    # PM #
    ######

    # pmra, pmdec = pmphi12(phi1, phi2, mu_phi1_cos_phi2_stream,
    #                       mu_phi2_stream, R_phi12_radec.T)

    # pmra_s5 = s5_data[:, 4:6]
    # pmdec_s5 = s5_data[:, 6:8]

    # pmra_s5 = s5_data[:, 4:6]
    # pmdec_s5 = s5_data[:, 6:8]

    chi2_pmra = 0.

    for counter in range(len(pm1_arr['phi1'])):
        t_var = pm1_arr['phi1'][counter]
        mask = np.where((phi1 > t_var - dL0 / 2.) * (phi1 < t_var + dL0 / 2.))[0]  # approx length fit with gmm
        if len(mask) > 5:

            # COMPUTE MAX LIKELIHOOD LOCATION OF STREAM OBSERVABLE
            phi1_data = phi1[mask]
            phi2_data = mu_phi1_cos_phi2_stream[mask]  # This needs to change for each observable  #########
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
            sigma_data = b_error  # sigma_estimate  # because comparing to individual data points, not binned fit

        # elif len(mask) == 1:
        #     continue
        else:
            chi2_pmra += 100
            continue

        # COMPARE SIMULATION TO DATA
        mu_obs = pm1_arr['pmphi1'][counter]
        sigma_obs = pm1_arr['epmphi1'][counter]
        sigma = (sigma_data**2. + sigma_obs**2.)**0.5
        chi2_pmra += np.log(np.sqrt(2. * np.pi * sigma**2.)) + 0.5 * (mu_data - mu_obs)**2. / (sigma)**2.

    chi2 += chi2_pmra

    #################

    chi2_pmdec = 0.

    for counter in range(len(pm2_arr['phi1'])):
        t_var = pm2_arr['phi1'][counter]
        mask = np.where((phi1 > t_var - dL0 / 2.) * (phi1 < t_var + dL0 / 2.))[0]  # approx length fit with gmm
        if len(mask) > 5:

            # COMPUTE MAX LIKELIHOOD LOCATION OF STREAM OBSERVABLE
            phi1_data = phi1[mask]
            phi2_data = mu_phi2_stream[mask]  # This needs to change for each observable  #########
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
            sigma_data = b_error  # sigma_estimate  # because comparing to individual data points, not binned fit

        # elif len(mask) == 1:
        #     continue
        else:
            chi2_pmdec += 100
            continue

        # COMPARE SIMULATION TO DATA
        mu_obs = pm2_arr['pmphi2'][counter]
        sigma_obs = pm2_arr['epmphi2'][counter]
        sigma = (sigma_data**2. + sigma_obs**2.)**0.5
        chi2_pmdec += np.log(np.sqrt(2. * np.pi * sigma**2.)) + 0.5 * (mu_data - mu_obs)**2. / (sigma)**2.

    chi2 += chi2_pmdec

    #########################
    ### Radial Velocities ###
    #########################

    # vr_s5 = s5_data[:, 2:4]

    chi2_vr = 0.

    if PLOTTING:
        vr_mu_data = []
        vr_mu_obs = []
        vr_sig_obs = []
        phi1_mu_obs = []

    for counter in range(len(rv_arr['phi1'])):
        t_var = rv_arr['phi1'][counter]
        mask = np.where((phi1 > t_var - dL0 / 2.) * (phi1 < t_var + dL0 / 2.))[0]
        if len(mask) > 5:

            # COMPUTE MAX LIKELIHOOD LOCATION OF STREAM OBSERVABLE
            phi1_data = phi1[mask]
            phi2_data = vr_gsr_stream[mask]
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
            sigma_data = b_error  # sigma_estimate  # because comparing to individual data points, not binned fit

            if PLOTTING:
                vr_mu_obs.append(rv_arr['rv_gsr'][counter])
                vr_sig_obs.append(rv_arr['erv'][counter])
                vr_mu_data.append(mu_data)
                phi1_mu_obs.append(t_var)

        # elif len(mask) == 1:
        #     continue
        else:
            chi2_vr += 100
            continue

        # COMPARE SIMULATION TO DATA
        mu_obs = rv_arr['rv_gsr'][counter]
        sigma_obs = rv_arr['erv'][counter]
        sigma = (sigma_data**2. + sigma_obs**2.)**0.5 #  + sigma_vr**2.)**0.5
        chi2_vr += np.log(np.sqrt(2. * np.pi * sigma**2.)) + 0.5 * (mu_data - mu_obs)**2. / (sigma)**2.

    chi2 += chi2_vr

    ############
    # DISTANCE #
    ############

    if DIST == True:
        print('Calculating chi2_dist...')

        chi2_dist = 0.
        for counter in range(len(dm_arr['phi1'])):
            t_var = dm_arr['phi1'][counter]
            mask = np.where((phi1 > t_var - dL0 / 2.) * (phi1 < t_var + dL0 / 2.))[0]
            if len(mask) > 5:

                # COMPUTE MAX LIKELIHOOD LOCATION OF STREAM OBSERVABLE
                phi1_data = phi1[mask]
                phi2_data = dist2mod(r_stream[mask])
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
                sigma_data = b_estimate # sigma_estimate  # because comparing to individual data points, not binned fit

            else:
                chi2_dist += 100
                continue

            # COMPARE SIMULATION TO DATA
            mu_obs = dm_arr['dm'][counter]
            sigma_obs = dm_arr['edm'][counter]
            sigma = (sigma_data**2. + sigma_obs**2.)**0.5
            chi2_dist += np.log(np.sqrt(2. * np.pi * sigma**2.)) + 0.5 * (mu_data - mu_obs)**2. / (sigma)**2.

        chi2 += chi2_dist
    else:
        print('Skipping chi2_dist...')
        chi2_dist = 0

        #################

    ##########################################################################

    if STREAM == 'Jhelum':
        fileout = open(OUTPUTDIR + 'save_chi2_%s_%i_%i_%s_%i.txt' %
                       (STREAM.replace(' ', '_'), JHELUM_POP, VERSION, PHI1, pid), 'a')
    else:
        fileout = open(OUTPUTDIR + 'save_chi2_%s_%i_%s_%i.txt' %
                       (STREAM.replace(' ', '_'), VERSION, PHI1, pid), 'a')

    print(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog,
          phi2_prog, M_LMC, sigma_track, sigma_vr, M_NFW, chi2_st, chi2_pmra, chi2_pmdec, chi2_vr, chi2_dist, chi2, file=fileout)

    fileout.close()

    if PLOTTING:
        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(16,16))
        fig.subplots_adjust(hspace=0)

        s5_data = np.loadtxt('../Chenab_S5.txt')
        pmra_s5 = s5_data[:, 4:6]
        pmdec_s5 = s5_data[:, 6:8]
        vr_s5 = s5_data[:, 2:4]
        phi1_s5, phi2_s5 = phi12_rotmat(s5_data[:, 0], s5_data[:, 1], R_phi12_radec)
        pmphi1_s5, pmphi2_s5 = pmphi12(s5_data[:, 0], s5_data[:, 1], pmra_s5[:, 0], pmdec_s5[:, 0], R_phi12_radec)
        pmerr_s5 = np.sqrt(pmra_s5[:, 1]**2 + pmdec_s5[:, 1]**2)

        import pandas as pd
        dist_data = pd.read_csv('../stream_distances.csv')
        dist_data = dist_data[dist_data['stream'] == STREAM]
        phi1_dist, phi2_dist = phi12_rotmat(dist_data['ra'].to_numpy(), dist_data['dec'].to_numpy(), R_phi12_radec)
        dm_dist = dist_data[['dm', 'dm_err']].to_numpy()
        type_dist = dist_data['tracer']

        ax[0].scatter(phi1_s5, phi2_s5, c='r', s=1)
        ax[1].errorbar(phi1_dist, dm_dist[:,0], yerr=dm_dist[:,1], c='r',marker='o',ecolor='r',markerfacecolor='r',capsize=3,linestyle='None',ms=1)
        ax[2].errorbar(phi1_s5, pmphi1_s5, yerr=pmerr_s5, c='r',marker='o',ecolor='r',markerfacecolor='r',capsize=3,linestyle='None',ms=1)
        ax[3].errorbar(phi1_s5, pmphi2_s5, yerr=pmerr_s5, c='r',marker='o',ecolor='r',markerfacecolor='r',capsize=3,linestyle='None',ms=1)
        # ax[4].errorbar(phi1_s5, vr_s5[:,0], yerr=vr_s5[:,1], c='r',marker='o',ecolor='r',markerfacecolor='r',capsize=3,linestyle='None',ms=1)

        ax[0].errorbar(track_arr['phi1'],track_arr['phi2'],yerr=track_arr['ephi2'],c='navy',marker='o',ecolor='navy',markerfacecolor='navy',capsize=3,linestyle='None',ms=1)
        ax[1].errorbar(dm_arr['phi1'],dm_arr['dm'],yerr=dm_arr['edm'],c='navy',marker='o',ecolor='navy',markerfacecolor='navy',capsize=3,linestyle='None',ms=1)
        ax[2].errorbar(pm1_arr['phi1'],pm1_arr['pmphi1'],yerr=pm1_arr['epmphi1'],c='navy',marker='o',ecolor='navy',markerfacecolor='navy',capsize=3,linestyle='None',ms=1)
        ax[3].errorbar(pm2_arr['phi1'],pm2_arr['pmphi2'],yerr=pm2_arr['epmphi2'],c='navy',marker='o',ecolor='navy',markerfacecolor='navy',capsize=3,linestyle='None',ms=1)
        ax[4].errorbar(rv_arr['phi1'],rv_arr['rv_gsr'],yerr=rv_arr['erv'],c='navy',marker='o',ecolor='navy',markerfacecolor='navy',capsize=3,linestyle='None',ms=1)

        fs=12
        ax[0].set_ylabel(r'$\phi_2$ (deg)',fontsize=fs)
        ax[1].set_ylabel(r'$m-M$',fontsize=fs)
        ax[2].set_ylabel(r'$\mu_{\phi_1}$ (mas/yr)',fontsize=fs)
        ax[3].set_ylabel(r'$\mu_{\phi_2}$ (mas/yr)',fontsize=fs)
        ax[4].set_ylabel(r'$v_{\rm gsr}$ (km/s)',fontsize=fs)
        ax[4].set_xlabel(r'$\phi_1$ (deg)',fontsize=12)
        
        ax[0].scatter(phi1, phi2, c='c', s=1, zorder=0, alpha=0.2)
        ax[1].scatter(phi1, dist2mod(r_stream), c='c', s=1, zorder=0, alpha=0.2)
        ax[2].scatter(phi1, mu_phi1_cos_phi2_stream, c='c', s=1, zorder=0, alpha=0.2)
        ax[3].scatter(phi1, mu_phi2_stream, c='c', s=1, zorder=0, alpha=0.2)
        ax[4].scatter(phi1, vr_gsr_stream, c='c', s=1, zorder=0, alpha=0.2)
        
        # RRLs
        orph_rrls = fitsio.open('../orph.fits')[1].data
        
        orph_phi1, orph_phi2 = phi12_rotmat(orph_rrls['ra'], orph_rrls['dec'], R_phi12_radec)
        # orph_mod = dist2mod(orph_rrls['heldist'])
        
        ax[0].scatter(orph_phi1, orph_phi2, c='k', s=5)
    
        plt.tight_layout()

        plt.savefig('../plots/%s_stream_%2f_%i.png' %
                    (STREAM.replace(' ', '_'), phi1_prog, pid), bbox_inches='tight')

    # print('chi2 shapes: ', chi2.size, chi2_st.size, chi2_pmra.size, chi2_pmdec.size, chi2_vr.size)
    if chi2 == 0. or chi2_st == 0. or chi2_pmra == 0. or chi2_pmdec == 0. or chi2_vr == 0.:
        return -np.inf
    else:
        return -chi2  # , -chi2_st, -chi2_pmra, -chi2_pmdec, -chi2_vr, -chi2_dist
