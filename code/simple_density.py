import numpy as np
import healpy as hp
import fitsio as fits
import glob
import matplotlib.pyplot as plt
import sys

#fnames_skim = glob.glob('/data/des81.b/data/tavangar/skim_1.1/*.fits')
fnames_skim = glob.glob('/data/des40.b/data/decals/dr8/south_skim/*.fits')

high_mag = np.float(sys.argv[1])

healpix = np.zeros(hp.nside2npix(256),dtype=np.int32)


for f in fnames_skim:
   data = fits.read(f)
   #star =  data['EXT_SOF']
   star = data['EXTENDED_CLASS']
   #g = data['SOF_PSF_MAG_CORRECTED_G']
   #r = data['SOF_PSF_MAG_CORRECTED_R']
   g = data['MAG_SFD_G']
   r = data['MAG_SFD_R']

   blue = g-r
   ra = data['RA']
   dec = data['DEC']
   print(f)
   for i in range(len(blue)):
      if blue[i] < 0.3 and star[i] == 0 and g[i] < high_mag:
         pix = hp.ang2pix(256, np.radians(-dec[i]+90.), np.radians(ra[i]))
         healpix[pix] += 1
np.save('decals/mag_cuts/star_cut_mod_{}.npy'.format(int(high_mag*10)), healpix)
#ebv = hp.udgrade(hp.readmap('/home/s1/tavangar/tav_simple/masking/ebv_sfd98_fullres_nside_4096_ring_equatorial.fits'), nside)
hp.mollzoom(healpix)
#hp.mollview(ebv)
plt.ion()
plt.show()
plt.savefig('mag_cuts_png/star_cut_mod_{}.png'.format(int(high_mag*10)))
