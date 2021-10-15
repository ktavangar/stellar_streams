from __future__ import print_function

import matplotlib.pyplot as plt

import numpy as np

import emcee

from scipy.interpolate import interp1d

import scipy.optimize as sopt

import chi2_jet_width as chi2

import sys

import os

from contextlib import closing
from multiprocessing import Pool


from stream_data import stream_phi120_pms, stream_dists, stream_vrs

STREAM = chi2.STREAM
JHELUM_POP = chi2.JHELUM_POP

print('Fitting stream: %s, Jhelum pop: %i, Version: %i, Phi1: %s' % (STREAM, JHELUM_POP, chi2.VERSION, chi2.PHI1))

# STREAM = 'ATLAS'
# STREAM = 'Phoenix'
# STREAM = 'Chenab'
# STREAM = 'Elqui'

if STREAM not in ['Orphan', 'Jet']:
    # phi1_pm_obs = 0.
    pm10_obs = stream_phi120_pms[STREAM]['pm1']
    pm20_obs = stream_phi120_pms[STREAM]['pm2']
    dist_obs = stream_dists[STREAM]
    vr0_obs = stream_vrs[STREAM]
    phi2_obs = 0.


if STREAM == 'Jet':
    if chi2.PHI1 == 'mid':
        pm1, pm2, vr, dist, phi2 = np.array([-1.22037387e+00, -5.22295001e-01,  2.91222307e+02,  3.14060933e+01, -1.33140912e-01])
        M_LMC = 15.



################################################################
# LIKELIHOOD CALL
################################################################

def dist2mod(distance):
    return 5. * (np.log10(np.array(distance) * 1.e3) - 1.)


def mod2dist(distance_modulus):
    return 10**(distance_modulus / 5. + 1.) / 1e3


def lnprior(vals):
    M_prog = 10**vals[0]
    # sigma_prog = vals[1]
    sigma_prog = 0.01

    if (M_prog >= 1e-7) & (M_prog <= 1e-3): # & (sigma_prog > 1e-4) & (sigma_prog < 1.0):
        return -np.log(M_prog)
    else:
        return -np.inf


def likelihood(vals):
    M_prog = 10**vals[0]
    # sigma_prog = vals[1]
    sigma_prog = 0.01

    lp = lnprior(vals)
    if not np.isfinite(lp):
        return -np.inf

    logL = lp + chi2.chi2_eval(pm1, pm2, vr, dist, phi2, M_LMC, M_prog=M_prog, sigma_prog=sigma_prog)

    if np.isnan(logL):
        return -np.inf
    else:
        return logL


nsteps = 2000
nthreads = 32
ndim, nwalkers = 1, 512


def rand_array():
    ret_array = np.zeros(ndim)
    ret_array[0] = np.random.uniform(-7, -3) # log10 M_prog
    # ret_array[1] = 10**np.random.uniform(-4, 0)
    return ret_array


def rand_array_walkers():
    return [rand_array() for i in range(nwalkers)]


###############################################################################

if __name__ == '__main__':
    RESTART = False
    if RESTART:
        # outfile = 'samples_Indus_29_mid_768347.h5'
        print('Restarting from %s...' % outfile)
        backend = emcee.backends.HDFBackend(outfile)
        with closing(Pool(processes=nthreads)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=likelihood, backend=backend, pool=pool)

            index = 0
            autocorr = np.empty(nsteps)
            old_tau = np.inf

            pos = backend.get_last_sample()

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

    else:
        ICs = rand_array_walkers()

        pid = os.getpid()
        outfile = 'samples_%s_%i_%s_width_%i.h5' % (STREAM.replace(' ', '_'), chi2.VERSION, chi2.PHI1, pid)

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

    f = open("samples_%s_%i_%s_width.dat" % (STREAM.replace(' ', '_'), chi2.VERSION,  chi2.PHI1), "w")

    print('#Acceptance fraction is {0:.3f}'.format(
        np.mean(sampler.acceptance_fraction)), file=f)

    lhood = sampler.lnprobability.reshape((-1, nwalkers * nsteps))

    for i in range(len(samples)):
        print(samples[i][0], lhood[0][i], file=f)

    f.close()
