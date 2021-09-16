import numpy as np
import healpy as hp
import fitsio as fits

def ang2pix(nside, lon, lat, nest=False):
    """
    Input (lon, lat) in degrees instead of (theta, phi) in radians
    """
    pix = []
    theta = np.radians(90. - lat)
    phi = np.radians(lon)
    for i in range(len(theta)):
        pix = np.append(pix,hp.ang2pix(nside, theta[i], phi, nest=nest))
    print(len(pix))
    pix = np.unique(pix)
    return pix

def get_files(ra, dec):
    ras = []
    decs = []
    for i in range(15):
        ras = np.append(ras, ra+i)
        ras = np.append(ras, ra-i)
        decs = np.append(decs, dec+i)
        decs = np.append(decs, dec-i)
    ind = ang2pix(32, ras, decs)
    print(ind)
    print(len(ind))
    filenames = []
    for i in range(len(ind)):
        filenames = np.append(filenames, '/data/des81.b/data/tavangar/south_skim/decals-dr8-sweep_0{}.fits'.format(int(ind[i])))
    return(filenames)
