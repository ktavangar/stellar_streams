#!/usr/bin/env python3
# https://stackoverflow.com/a/32297563/4075339

import numpy as np
from numpy.polynomial import polynomial

import scipy
import scipy.stats
from scipy.optimize import curve_fit

def polyfit2d(x, y, f, deg, var=True):
    """
    Fit a 2d polynomial.

    Parameters:
    -----------
    x : array of x values
    y : array of y values
    f : array of function return values
    deg : polynomial degree (length-2 list)

    Returns:
    --------
    c : polynomial coefficients
    """
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c, res, rank, s = np.linalg.lstsq(vander, f, rcond=None)
    return c.reshape(deg+1), res

def polyfuncdeg6(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, 
                 c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
                c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,
                c40, c41, c42, c43, c44, c45, c46, c47, c48):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 49
    '''
    (x,y) = xymesh
    
    coeffs = [[c0, c1, c2, c3, c4, c5, c6], 
                [c7, c8, c9, c10, c11, c12, c13], 
                [c14, c15, c16, c17, c18, c19, c20], 
                [c21, c22, c23, c24, c25, c26, c27], 
                [c28, c29, c30, c31, c32, c33, c34], 
                [c35, c36, c37, c38, c39, c40, c41],
                [c42, c43, c44, c45, c46, c47, c48]]
    
    func = polynomial.polyval2d(x, y, coeffs)
    return(np.ravel(func))



def polyfuncdeg5(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, 
                 c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
                c30, c31, c32, c33, c34, c35):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 36
    '''
    (x,y) = xymesh
    
    coeffs = [[c0, c1, c2, c3, c4, c5], 
                [c6, c7, c8, c9, c10, c11], 
                [c12, c13, c14, c15, c16, c17], 
                [c18, c19, c20, c21, c22, c23], 
                [c24, c25, c26, c27, c28, c29],
                [c30, c31, c32, c33, c34, c35]]
    
    func = polynomial.polyval2d(x, y, coeffs)
    return(np.ravel(func))

def polyfuncdeg4(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, 
                 c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                c20, c21, c22, c23, c24):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 25
    '''
    (x,y) = xymesh
    
    coeffs = [[c0, c1, c2, c3, c4],
               [c5, c6, c7, c8, c9], 
               [c10, c11,c12, c13, c14], 
               [c15, c16, c17, c18, c19],
               [c20, c21, c22, c23, c24]]
    
    func = polynomial.polyval2d(x, y, coeffs)
    return(np.ravel(func))

def polyfuncdeg3(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, 
                 c10, c11, c12, c13, c14, c15):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 16
    '''
    (x,y) = xymesh
    
    coeffs = [[c0, c1, c2, c3],
               [c4, c5, c6, c7], 
               [c8, c9,c10, c11], 
               [c12, c13, c14, c15]]
    
    func = polynomial.polyval2d(x, y, coeffs)
    return(np.ravel(func))

def polyfuncdeg2(xymesh, c0, c1, c2, c3, c4, c5, c6, c7, c8):
    '''
    xymesh: the independent variables
    coeffs: the coefficients that will be fitted, should have len 36
    '''
    (x,y) = xymesh
    
    coeffs = [[c0, c1, c2], [c3, c4, c5], [c6, c7, c8]]
    
    func = polynomial.polyval2d(x, y, coeffs)
    return(np.ravel(func))

def curvefit2d(deg, xymesh, f):
    '''
    func: function with the parameters to fit
    xymesh: x and y value meshgrid to input into func
    f: already flattened array of the data
    '''
    if deg == 2:
        func = polyfuncdeg2
    elif deg == 3:
        func = polyfuncdeg3
    elif deg == 4:
        func = polyfuncdeg4
    elif deg == 5:
        func = polyfuncdeg5
    elif deg == 6:
        func = polyfuncdeg6
    
    fit_params, cov_mat = curve_fit(func, xymesh, f)
    fit_errors = np.sqrt(np.diag(cov_mat))
    
    (x, y) = xymesh
    fit_residual = f - func(xymesh, *fit_params).reshape(f.shape)
    fit_Rsquared = 1 - np.var(fit_residual)/np.var(f)
    
    fit_params = fit_params.reshape(deg+1,deg+1)
    fit_errors = fit_errors.reshape(deg+1,deg+1)
    
    #print('Fit R-squared:', fit_Rsquared)
    #print('Fit Coefficients:', fit_params)
    #print('Fit errors:', fit_errors)
    
    return fit_params, fit_errors
   
