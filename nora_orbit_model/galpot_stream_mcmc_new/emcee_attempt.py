from __future__ import print_function

import matplotlib.pyplot as plt

import numpy as np

import emcee

from scipy.interpolate import interp1d

import scipy.optimize as sopt

# import chi2_newatlas as chi2
# import chi2_orphan as chi2
# import chi2_jet as chi2
# import chi2
# import chi2_chenab_orphan as chi2
# import chi2_chenab_splines as chi2
import chi2_kiyan_phoenix as chi2

import sys

import os

from contextlib import closing
from multiprocessing import Pool


#from stream_data import stream_phi120_pms, stream_dists, stream_vrs

STREAM = chi2.STREAM
JHELUM_POP = chi2.JHELUM_POP

print('Fitting stream: %s, Jhelum pop: %i, Version: %i, Phi1: %s' % (STREAM, JHELUM_POP, chi2.VERSION, chi2.PHI1))

# STREAM = 'ATLAS'
# STREAM = 'Phoenix'
# STREAM = 'Chenab'
# STREAM = 'Elqui'

#if STREAM not in ['Orphan', 'Jet', '300S']:
#    # phi1_pm_obs = 0.
#    pm10_obs = stream_phi120_pms[STREAM]['pm1']
#    pm20_obs = stream_phi120_pms[STREAM]['pm2']
#    dist_obs = stream_dists[STREAM]
#    vr0_obs = stream_vrs[STREAM]
#    phi2_obs = 0.


# ATLAS: 0, neg, pos
if STREAM == 'ATLAS':
    if chi2.PHI1 == 'mid':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = [-0.3968142624318181, -0.8757014083590925, -108.82023413736783, 20.086439549163458, 0.7357552547183849]
    elif chi2.PHI1 == 'neg':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = [-0.5425648465977314, -0.6111795288113088, -47.39015206250116, 23.92651678825912, -0.13282568723240526]
    elif chi2.PHI1 == 'pos':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = [-0.016005974256729304, -1.2012568175632488, -153.04580961460326, 17.403253612821764, 0.4771241768456409]
elif STREAM == 'Phoenix':
    if chi2.PHI1 == 'mid':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = [-0.8350584813742976, -2.5975585975585176, 48.856385899479065, 16.123534338890465, -0.07162302527671616]
    elif chi2.PHI1 == 'neg':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = [-0.5865300934377425, -2.646123636945348, 19.466507281728397, 15.193703188330662, 0.47570072552090104]
    elif chi2.PHI1 == 'pos':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = [-1.124042707438284, -2.589290457546308, 73.32444353729724, 16.868026061835547, -0.11481842620705818]
elif STREAM == 'Chenab':
    if chi2.PHI1 == 'mid':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 2.0939567958493757, -1.2399018885384299, -145.9129900413324, 40.286974127075474, 0.08488212823911338
    elif chi2.PHI1 == 'neg':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 1.5774584130475446, -0.9622461014386219, -170.22852967023152, 50.028141099489524, 0.059741915546388774
    elif chi2.PHI1 == 'pos':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 2.5575960954166246, -1.4857323550665225, -105.85403698164652, 34.35078592153441, -0.18692135340496624
    elif chi2.PHI1 == 20:
        print('Using Chenab phi1 = 20 params')
        # pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 2.8593288417242415, -1.6474119677832653, -70.17412613089597, 31.10455669703969, -0.45492455212359023-2
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 2.89857004e+00, -1.63334700e+00, -7.33712025e+01,  2.98410932e+01, -8.25844586e-01
    elif chi2.PHI1 == 40:
        print('Chenab phi1 = 40 params')
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 3.269459234956346, -1.8867747223425066, 13.112009602974725, 26.358499552461225, -1.0154598696582007
    elif np.abs(float(chi2.PHI1) + 60) < 1:
        print('phi1_prog = -60!')
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 2.1318, 1.1075, -147.3696, 38.3922, 1.7209
    elif np.abs(float(chi2.PHI1) - 60) < 10:
        print('phi1_prog ~ 60')
        # pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 4.1540000617863635, -2.0036275671988446, 70.80933798279804, 19.349910609817545, -2.420363923153127
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 3.731003072313108, -2.127294979154763, 101.123115, 20.4040969, -0.60332853663272357
elif STREAM == 'Elqui':
    if chi2.PHI1 == 'mid':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -0.5651822453274788, -0.2642616184962703, -54.94356040441061, 62.05684157271919, 0.12055597258378052
    elif chi2.PHI1 == 'neg':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -0.507992188580888, -0.20012959922080875, 42.26848609774498, 69.92349797876393, -0.37379132837334417
    elif chi2.PHI1 == 'pos':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -0.6342823176439477, -0.3957227157757078, -135.4014496412144, 50.79607853899732, -1.0457372750437388
elif STREAM == 'Indus':
    if chi2.PHI1 == 'mid':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -6.0611598948987755, -1.2654538013324257, -53.579643110960795, 15.184917908017642, -0.22740858828769578
    elif chi2.PHI1 == 'neg':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -6.565872318924768, -1.7036388990254623, -2.4518583540618377, 13.601138201359907, 0.7057782014938413
    elif chi2.PHI1 == 'pos':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -5.385278112180077, -0.9158334244003213, -97.53982403835346, 17.15032402352689, -2.066995172085992
elif STREAM == 'Jhelum':
    if JHELUM_POP == 1:
        if chi2.PHI1 == 'mid':
            pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -7.124463557166753, -3.1487489722033155, -8.549264503152312, 12.771828626771804, -0.07275312819048993
    elif JHELUM_POP == 2:
        if chi2.PHI1 == 'mid':
            pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -7.98939327623298, -3.9899574638718964, -41.20215173660577, 12.395324680877694, -0.06474213168187315
elif STREAM == 'TucIII':
    if chi2.PHI1 == 'mid':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 0.22117817, -1.6280914, -100, 25.0, 0.0
elif STREAM == 'Jet':
    if chi2.PHI1 == 'mid':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -1.22037387e+00, -5.22295001e-01,  2.91222307e+02,  3.14060933e+01, -1.33140912e-01  # -0.2, -0.2, 270.0, 30.0, 0.0
    elif np.abs(float(chi2.PHI1) + 20) < 0.01:
        print('Using Jet phi1 = -20 params!')
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -0.6872283750519611, -0.5491032196716242, 195.85519520338457, 33.324382027221446, -5.66286198
    elif np.abs(float(chi2.PHI1) - 20) < 0.01:
        print('Using Jet phi1=20 params!')
        # pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -1., 0., 330., 23., 0.
    elif np.abs(float(chi2.PHI1) - 15) < 0.01:
        print('Using Jet phi1=15 params!')
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = -2.1655875, -0.338259, 280., 27., -0.8
elif STREAM == '300S':
    if chi2.PHI1 == 'mid':
        pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 2.8, 3.5,  300, 18, 0.5


###################### CHANGE THIS #
# pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = 3.829116452292585, 1.9451568388805622, 101.123115, 20.4040969, -1.42827013
# 4.494947706193628, 2.039448407400339, 51.582322977964935, 17.215475721632853, -1.3802989286023066
# 3.731003072313108, -2.127294979154763, 101.123115, 20.4040969, -0.60332853663272357
# print('Orphan phi1=6.34!!!!')

print('pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs = ', pm10_obs, pm20_obs, vr0_obs, dist_obs, phi2_obs)

distance_sigma = 2.
mod_sigma = 0.2
if chi2.PHI1 != 'mid':
    phi2_lim = 10.0
else:
    phi2_lim = 1.0
phi2_lim = 10.0
print('phi2_lim = %.2f' % phi2_lim)

# if STREAM == 'Elqui' and chi2.PHI1 == 'mid':
#     dist_obs = 50.0
#     mod_sigma = 0.4
#     print('Elqui dist: %.2f, %.2f' %(dist_obs, mod_sigma))

################################################################
# LIKELIHOOD CALL
################################################################

# chi2_eval(mu_alpha,mu_delta,rv,dist,M_NFW,rs_NFW,q_NFW,theta_NFW,phi_NFW)
# def
# lnprior(mu_alpha,mu_delta,rv,dist,M_NFW,rs_NFW,q_NFW,M_LMC,mu_alpha_lmc,mu_delta_lmc,rv_lmc,dist_lmc,R0,Mprog):


def dist2mod(distance):
    return 5. * (np.log10(np.array(distance) * 1.e3) - 1.)


def mod2dist(distance_modulus):
    return 10**(distance_modulus / 5. + 1.) / 1e3


def lnprior(vals):
    mu_phi1_prog = vals[0]
    mu_phi2_prog = vals[1]
    rv_prog = vals[2]
    dist_prog = vals[3]
    phi2_prog = vals[4]

    mod_prog = dist2mod(dist_prog)

    if len(vals) == 7:
        sigma_extra = vals[5]
        sigma_vr = vals[6]
        M_LMC = 15.
        if sigma_extra < 0 or sigma_extra > 2 or sigma_vr < 0 or sigma_vr > 20:
            return -np.inf
    elif len(vals) < 6:
        M_LMC = 15.
    elif len(vals) == 10:
        sigma_extra = 0.
        sigma_vr = 0.
        M_LMC = vals[5]
    else:
        M_LMC = vals[5]
        if len(vals) > 6:
            sigma_extra = vals[6]
            sigma_vr = vals[7]
            if sigma_extra < 0 or sigma_extra > 2 or sigma_vr < 0 or sigma_vr > 20:
                return -np.inf

    mod_prog = dist2mod(dist_prog)
    mod_obs = dist2mod(dist_obs)

    lnp = 0

    if len(vals) > 8:
        mu_alpha_lmc = vals[-4]
        mu_delta_lmc = vals[-3]
        rv_lmc = vals[-2]
        dist_lmc = vals[-1]

        lnp += -((mu_alpha_lmc - 1.91) / 0.02)**2. / 2.
        lnp += -((mu_delta_lmc - 0.229) / 0.047)**2. / 2.
        lnp += -((rv_lmc - 262.2) / 3.4)**2. / 2.
        lnp += -((dist_lmc - 49970.) / 1126.)**2. / 2.

    if M_LMC > 30. or M_LMC < 2:
        return -np.inf
    elif np.abs(rv_prog) < 500. and np.abs(mu_phi1_prog) < 10. and np.abs(mu_phi2_prog) < 10. and np.abs(phi2_prog) <= phi2_lim and STREAM in ['Indus', 'Jhelum', 'Chenab']:
        lnp += -np.log(M_LMC)
    elif np.abs(rv_prog) < 500. and np.abs(mu_phi1_prog) < 5. and np.abs(mu_phi2_prog) < 5. and np.abs(phi2_prog) <= phi2_lim:
        lnp += -np.log(M_LMC)
    else:
        return -np.inf

    # if STREAM == '300S':
    #     lnp += -((dist_prog - 18) / distance_sigma)**2. / 2.
    
    # FOR KIYAN PHOENIX
    if chi2.PHI1 != 'mid':
        return lnp
    #lnp += -((mod_prog - dist2mod(stream_dists[STREAM])) / mod_sigma)**2. / 2.
    lnp += -((mod_prog - 16.19) / 1)**2. / 2.

    return lnp


def likelihood(vals):
    mu_phi1_prog = vals[0]
    mu_phi2 = vals[1]
    rv_prog = vals[2]
    dist_prog = vals[3]
    phi2_prog = vals[4]

    if len(vals) < 6:
        M_LMC = 15.
        sigma_extra = 0.
        sigma_vr = 0.

    elif len(vals) == 7:
        sigma_extra = vals[5]
        sigma_vr = vals[6]
        M_LMC = 15.0
    elif len(vals) == 10:
        sigma_extra = 0
        sigma_vr = 0
        M_LMC = vals[5]
    else:
        M_LMC = vals[5]

        if len(vals) > 6:
            sigma_extra = vals[6]
            sigma_vr = vals[7]
        else:
            sigma_extra = 0
            sigma_vr = 0

    if len(vals) > 8:
        mu_alpha_lmc = vals[-4]
        mu_delta_lmc = vals[-3]
        rv_lmc = vals[-2]
        dist_lmc = vals[-1]
    else:
        mu_alpha_lmc = np.random.normal(1.91, 0. * 0.02)
        mu_delta_lmc = np.random.normal(0.229, 0. * 0.047)
        rv_lmc = np.random.normal(262.2, 0. * 3.4)
        dist_lmc = np.random.normal(49970., 0. * 1126.)

    lp = lnprior(vals)
    if not np.isfinite(lp):
        return -np.inf

    logL = lp + chi2.chi2_eval(mu_phi1_prog, mu_phi2,
                               rv_prog, dist_prog, phi2_prog, M_LMC, sigma_extra, sigma_vr, mu_alpha_lmc, mu_delta_lmc, rv_lmc, dist_lmc)

    if np.isnan(logL):
        return -np.inf
    else:
        return logL

nsteps = 2000
nthreads = 64
ndim, nwalkers = 7, 256
# ndim, nwalkers = 5, 256
# ndim, nwalkers = 12, 256
# ndim, nwalkers = 6, 512
# ndim, nwalkers = 8, 512
# ndim, nwalkers = 10, 128

# emcee_ICs = np.array([-2.235,-2.228,-57.4,23600.,80.,16.,1.,1.,np.pi/2.])

# ICs = [emcee_ICs*np.random.normal(1.,0.1)+np.random.normal(0.0,0.05,9) for i in range(nwalkers)]


def rand_array():
    mod_obs = dist2mod(dist_obs)

    ret_array = np.zeros(ndim)
    if pm10_obs < 0:
        ret_array[0] = np.random.uniform(
            pm10_obs - 0.2, np.minimum(pm10_obs + 0.2, 0))
    elif pm10_obs > 0:
        ret_array[0] = np.random.uniform(
            np.maximum(pm10_obs - 0.2, 0), pm10_obs + 0.2)

    ret_array[1] = np.random.uniform(pm20_obs - 0.2, pm20_obs + 0.2)
    if vr0_obs == 0:
        ret_array[2] = np.random.uniform(-500., 500.)
    else:
        ret_array[2] = np.random.normal(vr0_obs, 10.)
    ret_array[3] = mod2dist(np.random.normal(mod_obs, 0.5))
    ret_array[4] = np.random.normal(phi2_obs, 0.5)
    # if chi2.PHI1 != 'mid':
    #     print('Broadening initials...')
    #     ret_array[0] = np.random.normal(pm10_obs, 1.0)
    #     ret_array[1] = np.random.normal(pm20_obs, 1.0)
    #     ret_array[2] = np.random.normal(vr0_obs, 50.)
    #     ret_array[3] = mod2dist(np.random.normal(mod_obs, 1.0))
    #     ret_array[4] = np.random.normal(phi2_obs, 2.0)
    # ret_array[4] = np.random.uniform(-1., 1.)

    if ndim == 7:
        ret_array[5] = np.random.uniform(0, 0.5)
        ret_array[6] = np.random.uniform(0, 10)
        return ret_array

    else:
        if ndim > 5:
            ret_array[5] = 10**np.random.uniform(np.log10(2), np.log10(30))

        if ndim > 6:
            if ndim == 10:
                pass
            ret_array[6] = np.random.uniform(0, 0.5)
            ret_array[7] = np.random.uniform(0, 10)

    if ndim > 8:
        ret_array[-4] = np.random.normal(1.91, 0.02)
        ret_array[-3] = np.random.normal(0.229, 0.047)
        ret_array[-2] = np.random.normal(262.2, 3.4)
        ret_array[-1] = np.random.normal(49970., 1126.)

    return ret_array


def rand_array_walkers():
    return [rand_array() for i in range(nwalkers)]


###############################################################################

if __name__ == '__main__':
    RESTART = False
    if RESTART:
        outfile = 'samples_Chenab_64_6.34_49461.h5'
        print('Restarting from %s...' % outfile)
        backend = emcee.backends.HDFBackend(outfile)
        with closing(Pool(processes=nthreads)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=likelihood, backend=backend, pool=pool)

            index = 0
            autocorr = np.empty(nsteps)
            old_tau = np.inf

            pos = backend.get_last_sample()

            chains = backend.get_chain(discard=0, flat=False)
            nsteps_0 = chains.shape[0]

            for sample in sampler.sample(pos, iterations=nsteps - nsteps_0, progress=True, store=True):
                if sampler.iteration % 100:
                    continue

                # get autocorrelation time
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau

            pool.terminate()

            # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend)

    else:
        ICs = rand_array_walkers()

        pid = os.getpid()
        if STREAM == 'Jhelum':
            outfile = 'samples_%s_%i_%i_%s_%i.h5' % (
                STREAM.replace(' ', '_'), JHELUM_POP, chi2.VERSION, chi2.PHI1, pid)
        else:
            outfile = 'samples_%s_%i_%s_%i.h5' % (STREAM.replace(' ', '_'), chi2.VERSION, chi2.PHI1, pid)

        backend = emcee.backends.HDFBackend(outfile)
        backend.reset(nwalkers, ndim)

        with closing(Pool(processes=nthreads)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=likelihood, backend=backend, pool=pool)
            pos, prob, state = sampler.run_mcmc(ICs, 200)
            sampler.reset()

            index = 0
            autocorr = np.empty(nsteps)
            old_tau = np.inf

            # store=True ?
            for sample in sampler.sample(pos, iterations=nsteps, progress=True, store=True):
                if sampler.iteration % 100:
                    continue

                # get autocorrelation time
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau

            pool.terminate()

    #######################

    samples = sampler.chain[:, :, :].reshape((-1, ndim))

    f = open("samples_%s_%i_%s.dat" % (STREAM.replace(' ', '_'), chi2.VERSION, chi2.PHI1), "w")

    print('#Acceptance fraction is {0:.3f}'.format(
        np.mean(sampler.acceptance_fraction)), file=f)

    lhood = sampler.lnprobability.reshape((-1, nwalkers * nsteps))

    for i in range(len(samples)):
        print(samples[i][0], samples[i][1], samples[i][2],
              samples[i][3], samples[i][4], samples[i][5], lhood[0][i], file=f)

    f.close()
