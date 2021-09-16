#!/usr/bin/env python
"""
Pseudo-fracdet map
"""
__author__ = "Sidney Mau"

import os
import glob
import yaml
import numpy as np
import healpy as hp
import fitsio as fits

############################################################

infiles = glob.glob ('/data/des81.b/data/tavangar/skim/*.fits')

nside = 256
pix = []
for infile in infiles:
    print('loading {}'.format(infile))
    data = fits.read(infile, columns=['RA','DEC'])
    p = hp.ang2pix(nside, data['RA'], data['DEC'], lonlat=True)
    pix.append(np.unique(p))

print('Constructing map')
pix = np.concatenate(pix)
pix = np.unique(pix)
coverage_map = np.tile(hp.UNSEEN, hp.nside2npix(nside))
coverage_map[pix] = 1

print('Writing output')
result = 'des_pseudo_fracdet.fits.gz'
hp.write_map(result, coverage_map)
