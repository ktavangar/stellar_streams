import numpy as np
import matplotlib.pyplot as plt
import os

import astropy.units as u
from astropy.coordinates import SkyCoord

import numpy
from cv_coord import cv_coord

import scipy.optimize as sopt

import gal_uvw

import vcirc

import scipy.interpolate as sinterp

import astropy.io.fits as afits
import scipy

from rotation_matrix import phi12_rotmat, pmphi12

M_NFW = 80.
q_NFW = 1.
rs_NFW = 16.

phi1_prog = 0.
phi2_prog = 0.

mu_phi2_prog = 0.

dist_prog = 19.1
rv_prog = 50.

R_phi12_radec = np.array([[0.5964467, 0.27151332, -0.75533559],
                         [-0.48595429, -0.62682316, -0.60904938],
                         [0.63882686, -0.73032406, 0.24192354]])

a_g = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                [-0.8734370902, -0.4448296300, -0.1980763734], 
                [-0.4838350155, 0.7469822445, +0.4559837762]])

G = 43007.105731706317

l_lmc = 280.4652*np.pi/180.
b_lmc = -32.8884*np.pi/180.

mcmillan = np.genfromtxt('ProductionRunBig2_Nora_10.tab')
mcm = mcmillan[9]

fileout = open('pot/PJM17_{0}.Tpot'.format(0),'w')

print >> fileout, 4
print >> fileout, '{0:.5e} {1:.5f} {2:.5f} {3} {4}'.format(mcm[0],mcm[1],mcm[2],mcm[3],mcm[4])
print >> fileout, '{0:.5e} {1:.5f} {2:.5f} {3} {4}'.format(mcm[5],mcm[6],mcm[7],mcm[8],mcm[9])
print >> fileout, '5.31319e+07 7 -0.085 4 0'
print >> fileout, '2.17995e+09 1.5 -0.045 12 0'

print >> fileout, 2
print >> fileout, '{0:.5e} {1:.5f} {2:.5f} {3} {4} {5}'.format(mcm[20],mcm[21],mcm[22],mcm[23],mcm[24],mcm[25])
print >> fileout, '{0:.5e} {1:.5f} {2:.5f} {3} {4} {5}'.format(mcm[26],mcm[27],mcm[28],mcm[29],mcm[30],mcm[31])

Usun = mcm[32]*1000.
Vsun = mcm[33]*1000.
Wsun = mcm[34]*1000.
R0 = mcm[-6]
V0 = mcm[-5]

M200 = 4.*np.pi*mcm[26]*mcm[30]**3.*(np.log(1.+mcm[-9]/mcm[30])-mcm[-9]/(mcm[-9]+mcm[30]))/(1.e10)

c200 = mcm[-9]/mcm[30]
rs = mcm[30]

fileout.close()

def chi2_eval(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog,dist_prog,phi2_prog):

    mu_alpha_lmc = np.random.normal(1.91,0.*0.02)
    mu_delta_lmc = np.random.normal(0.229,0.*0.047)
    rv_lmc = np.random.normal(262.2,0.*3.4)
    dist_lmc = np.random.normal(49970.,0.*1126.)

    M_LMC = 15.

    if M_LMC > 2.:
        rs_LMC = np.sqrt(G*M_LMC*8.7/91.7**2.)-8.7
    else:
        rs_LMC = np.sqrt(G*2.*8.7/91.7**2.)-8.7

    tmax = 3.
    Mprog = 2.e-6

    pid = os.getpid()
    
    lhood = chi2_worker(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog, phi2_prog,  mu_alpha_lmc, mu_delta_lmc, rv_lmc, dist_lmc, M_LMC,rs_LMC, Mprog, tmax, pid)
    return lhood
def chi2_worker(mu_phi1cosphi2_prog, mu_phi2_prog, rv_prog, dist_prog, phi2_prog, mu_alpha_lmc, mu_delta_lmc, rv_lmc, dist_lmc, M_LMC, rs_LMC, Mprog, tmax, pid):

    vec_phi12_prog = np.array([np.cos(phi1_prog*np.pi/180.)*np.cos(phi2_prog*np.pi/180.),np.sin(phi1_prog*np.pi/180.)*np.cos(phi2_prog*np.pi/180.),np.sin(phi2_prog*np.pi/180.)])

    vec_radec_prog = np.linalg.solve(R_phi12_radec,vec_phi12_prog)
    
    ra_prog = np.arctan2(vec_radec_prog[1],vec_radec_prog[0])*180./np.pi
    dec_prog = np.arcsin(vec_radec_prog[2]/np.linalg.norm(vec_radec_prog))*180./np.pi

    gc_prog = SkyCoord(ra=ra_prog*u.degree,dec=dec_prog*u.degree,frame='icrs')
    
    l_prog = np.array(gc_prog.galactic.l)
    b_prog = np.array(gc_prog.galactic.b)

    #mu_phi1cosphi2_prog = -12.4
    #mu_phi2_prog = -2.9

    vlsr = np.array([Usun,Vsun+V0,Wsun])
    
    M_UVW_muphi12_prog = np.array([[np.cos(phi1_prog*np.pi/180.)*np.cos(phi2_prog*np.pi/180.),-np.sin(phi1_prog*np.pi/180.),-np.cos(phi1_prog*np.pi/180.)*np.sin(phi2_prog*np.pi/180.)],[np.sin(phi1_prog*np.pi/180.)*np.cos(phi2_prog*np.pi/180.),np.cos(phi1_prog*np.pi/180.),-np.sin(phi1_prog*np.pi/180.)*np.sin(phi2_prog*np.pi/180.)],[np.sin(phi2_prog*np.pi/180.),0.,np.cos(phi2_prog*np.pi/180.)]])

    k_mu = 4.74047

    uvw_stationary = -vlsr

    vec_vr_muphi1_muphi2_stationary = np.dot(M_UVW_muphi12_prog.T,np.dot(R_phi12_radec,np.dot(a_g,uvw_stationary)))
    vec_vr_muphi1_muphi2_stationary[0] = 0. # no correction for radial velocity, i want our radial velocity to be the los one

    vec_vr_muphi1_muphi2_prog = np.array([rv_prog,k_mu*dist_prog*mu_phi1cosphi2_prog,k_mu*dist_prog*mu_phi2_prog]) #+ vec_vr_muphi1_muphi2_stationary
    
    vx_prog,vy_prog,vz_prog = np.dot(a_g.T,np.dot(R_phi12_radec.T,np.dot(M_UVW_muphi12_prog,vec_vr_muphi1_muphi2_prog))) + vlsr

    x_prog, y_prog, z_prog = np.array([-R0,0.,0.])+dist_prog*np.array([np.cos(l_prog*np.pi/180.)*np.cos(b_prog*np.pi/180.),np.sin(l_prog*np.pi/180.)*np.cos(b_prog*np.pi/180.),np.sin(b_prog*np.pi/180.)])

    vx,vy,vz = vx_prog,vy_prog,vz_prog
    
    x,y,z = x_prog,y_prog,z_prog

    gc = SkyCoord(b=b_lmc*u.radian,l=l_lmc*u.radian,frame='galactic')
    
    x_lmc,y_lmc,z_lmc = np.array([-R0,0.,0.])+dist_lmc/1000.*np.array([np.cos(l_lmc)*np.cos(b_lmc),np.sin(l_lmc)*np.cos(b_lmc),np.sin(b_lmc)])

    vx_lmc,vy_lmc,vz_lmc = gal_uvw.gal_uvw(distance=dist_lmc,ra=np.array(gc.icrs.ra),dec=np.array(gc.icrs.dec),lsr=np.array([-Usun,0.,Wsun]),pmra=mu_alpha_lmc,pmdec=mu_delta_lmc,vrad=rv_lmc)

    vy_lmc += Vsun+V0
    vx_lmc = -vx_lmc

    os.system('./a.out {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19}'.format(x,y,z,vx,vy,vz,M_LMC,rs_LMC,x_lmc,y_lmc,z_lmc,vx_lmc,vy_lmc,vz_lmc,Mprog,tmax,M200,rs,c200,pid))
              
    data = np.genfromtxt('final_stream_{0}.txt'.format(pid))

    
    pos = data[:,:3]
    vel = data[:,3:6]
    
    pos = pos - np.array([-R0,0.,0.])
        
    theta = np.arctan2(pos[:,1],pos[:,0])
    theta = np.mod(theta,2.*np.pi)
    theta[theta > np.pi] -= 2.*np.pi
    phi = np.arcsin(pos[:,2]/(pos[:,0]**2.+pos[:,1]**2.+pos[:,2]**2.)**0.5)

    l = 180./np.pi*theta
    b = 180./np.pi*phi

    gc = SkyCoord(l=l*u.degree,b=b*u.degree,frame='galactic')

    vec_radec = np.array([np.cos(gc.icrs.ra)*np.cos(gc.icrs.dec),np.sin(gc.icrs.ra)*np.cos(gc.icrs.dec),np.sin(gc.icrs.dec)])

    vec_phi12 = np.dot(R_phi12_radec,vec_radec).T

    phi1 = np.arctan2(vec_phi12[:,1],vec_phi12[:,0])*180./np.pi
    phi2 = np.arcsin(vec_phi12[:,2])*180./np.pi
    
    vel -= vlsr

    alpha = np.array(gc.icrs.ra)*np.pi/180.
    delta = np.array(gc.icrs.dec)*np.pi/180.

    r_stream = np.sum(pos[:,axis]**2. for axis in range(3))**0.5

    R_phi12_a_g = np.dot(R_phi12_radec,a_g)

    vr_stream = np.sum((np.cos(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*R_phi12_a_g[0,axis]+np.sin(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*R_phi12_a_g[1,axis]+np.sin(phi2*np.pi/180.)*R_phi12_a_g[2,axis])*vel[:,axis] for axis in range(3))

    mu_phi1_cos_phi2_stream = 1./(k_mu*r_stream)*np.sum( (-np.sin(phi1*np.pi/180.)*R_phi12_a_g[0,axis]+np.cos(phi1*np.pi/180.)*R_phi12_a_g[1,axis])*vel[:,axis] for axis in range(3))

    mu_phi2_stream = 1./(k_mu*r_stream)*np.sum( (-np.cos(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*R_phi12_a_g[0,axis] - np.sin(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*R_phi12_a_g[1,axis] + np.cos(phi2*np.pi/180.)*R_phi12_a_g[2,axis])*vel[:,axis] for axis in range(3))

    nvlsr = -vlsr

    mu_phi1_cos_phi2_stream_corr =  mu_phi1_cos_phi2_stream - 1./(k_mu*r_stream)*np.sum( (-np.sin(phi1*np.pi/180.)*R_phi12_a_g[0,axis]+np.cos(phi1*np.pi/180.)*R_phi12_a_g[1,axis])*nvlsr[axis] for axis in range(3))

    mu_phi2_stream_corr = mu_phi2_stream - 1./(k_mu*r_stream)*np.sum( (-np.cos(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*R_phi12_a_g[0,axis] - np.sin(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*R_phi12_a_g[1,axis] + np.cos(phi2*np.pi/180.)*R_phi12_a_g[2,axis])*nvlsr[axis] for axis in range(3))
    
    data_S5 = np.genfromtxt('../../../../../Data/S5/phoenix_zhang_31102019.csv',dtype=float,delimiter=',',names=True)

    phi1_rv,phi2_rv = phi12_rotmat(data_S5['ra'],data_S5['dec'],R_phi12_radec)
    
    chi2 = 0.
    dof = 0.
    chi2_st = 0.

    dL0 = 1.
    
    def return_mlhood(phi1_data,phi2_data,phi1_mid):
        def lhood(args):
            a_guess = args[0]
            b_guess = args[1]
            sigma_guess = args[2]
            sigma_eff = (sigma_guess**2.)**0.5
            phi2_guess = a_guess*(phi1_data-phi1_mid)+b_guess
            return np.sum(np.log(sigma_eff)+(phi2_data-phi2_guess)**2./(2.*sigma_eff**2.))

        def dlhood(args):
            a_guess = args[0]
            b_guess = args[1]
            sigma_guess = args[2]
            sigma_eff = (sigma_guess**2.)**0.5
            phi2_guess = a_guess*(phi1_data-phi1_mid)+b_guess
            dlogLda = -np.sum((phi2_data-phi2_guess)/(sigma_eff**2.)*(phi1_data-phi1_mid))
            dlogLdb = -np.sum((phi2_data-phi2_guess)/(sigma_eff**2.))
            dlogLdsigma = np.sum(1./sigma_guess - (phi2_data-phi2_guess)**2./(sigma_guess**3.))
            return np.array([dlogLda,dlogLdb,dlogLdsigma])
            
        
        result = sopt.fmin_bfgs(lhood,[0.,np.mean(phi2_data),np.std(phi2_data)],fprime=dlhood,full_output=True,disp=False)
            
        a_estimate = result[0][0]
        a_error = scipy.linalg.sqrtm(result[3])[0][0]
        
        b_estimate = result[0][1]
        b_error = scipy.linalg.sqrtm(result[3])[1][1]
        
        sigma_estimate = result[0][2]
        sigma_error = scipy.linalg.sqrtm(result[3])[2][2]
        
        mu_data = b_estimate
        sigma_data = sigma_estimate

        return mu_data,sigma_data
    
    for counter in range(len(phi1_rv)):
        t_var = phi1_rv[counter]

        mask = np.where((phi1 > t_var-dL0/2.)*(phi1 < t_var+dL0/2.))[0]
        if len(mask) > 5:
            phi1_data = phi1[mask]
            phi2_data = phi2[mask]
            phi1_mid = t_var
            mu_data,sigma_data = return_mlhood(phi1_data,phi2_data,phi1_mid)

        else:
            chi2_st += 100.
            continue

        mu_obs = phi2_rv[counter]
        sigma_obs = 0.
        sigma = (sigma_data**2.+sigma_obs**2.)**0.5
        chi2_st += np.log(np.sqrt(2.*np.pi*sigma**2.))+0.5*(mu_data-mu_obs)**2./(sigma)**2.
    
    chi2 += chi2_st

    chi2_vr = 0.
    
    pmra,pmdec = pmphi12(phi1,phi2,mu_phi1_cos_phi2_stream,mu_phi2_stream,R_phi12_radec.T)

    vr_sim = np.zeros(len(phi1_rv))
    
    for counter in range(len(phi1_rv)):
        t_var = phi1_rv[counter]
        
        mask = np.where((phi1 > t_var-dL0/2.)*(phi1 < t_var+dL0/2.))[0]
        if len(mask) > 5:
            phi1_data = phi1[mask]
            phi2_data = vr_stream[mask]
            phi1_mid = t_var
            mu_data,sigma_data = return_mlhood(phi1_data,phi2_data,phi1_mid)
        else:
            chi2_vr += 100.
            continue

        mu_obs = data_S5['rv'][counter]
        sigma_obs = data_S5['rv_error'][counter]
        sigma = (sigma_data**2.+sigma_obs**2.)**0.5

        chi2_vr += np.log(np.sqrt(2.*np.pi*sigma**2.))+0.5*(mu_data-mu_obs)**2./(sigma)**2.

        vr_sim[counter] = mu_data
        
    chi2 += chi2_vr

    chi2_pmra = 0.
    
    for counter in range(len(phi1_rv)):
        t_var = phi1_rv[counter]

        mask = np.where((phi1 > t_var-dL0/2.)*(phi1 < t_var+dL0/2.))[0]
        if len(mask) > 5:
            phi1_data = phi1[mask]
            phi2_data = pmra[mask]
            phi1_mid = t_var
            mu_data,sigma_data = return_mlhood(phi1_data,phi2_data,phi1_mid)
        else:
            chi2_vr += 100.
            continue

        mu_obs =  data_S5['pmra'][counter]
        sigma_obs = data_S5['pmra_error'][counter]
        sigma = (sigma_data**2.+sigma_obs**2.)**0.5
        chi2_pmra += np.log(np.sqrt(2.*np.pi*sigma**2.))+0.5*(mu_data-mu_obs)**2./(sigma)**2.


    chi2 += chi2_pmra

    chi2_pmdec = 0.
    
    for counter in range(len(phi1_rv)):
        t_var = phi1_rv[counter]

        mask = np.where((phi1 > t_var-dL0/2.)*(phi1 < t_var+dL0/2.))[0]
        if len(mask) > 5:
            phi1_data = phi1[mask]
            phi2_data = pmdec[mask]
            phi1_mid = t_var
            mu_data,sigma_data = return_mlhood(phi1_data,phi2_data,phi1_mid)
        else:
            chi2_vr += 100.
            continue

        mu_obs = data_S5['pmdec'][counter]
        sigma_obs = data_S5['pmdec_error'][counter]
        sigma = (sigma_data**2.+sigma_obs**2.)**0.5

        chi2_pmdec += np.log(np.sqrt(2.*np.pi*sigma**2.))+0.5*(mu_data-mu_obs)**2./(sigma)**2.

    chi2 += chi2_pmdec
    
    fileout = open('save_chi2_{0}.txt'.format(pid),'a')

    print >> fileout, mu_phi1cosphi2_prog,mu_phi2_prog,rv_prog,dist_prog,phi2_prog, chi2_st, chi2_vr, chi2_pmra, chi2_pmdec, chi2

    #print >> fileout, mu_alpha,mu_delta,rv,dist,M_NFW,rs_NFW,q_NFW,M_LMC,mu_alpha_lmc,mu_delta_lmc,rv_lmc,dist_lmc,R0,Mprog, chi2_vr, chi2_st, chi2, dt_strip

    fileout.close()

    if chi2 == 0.:
        return -np.inf
    else:
        return -chi2
