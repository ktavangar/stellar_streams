import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import ugali
from ugali.utils import stats
import pystan
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import importlib
sys.path.append('/data/des81.b/data/tavangar/streams/model_code')

test_extract = 1
stream = 'atlas_sergey'
print('open model')
with open("phoenix_models/{}_model_fit_test{}.pickle".format(stream, test_extract), "rb") as f:
    data_dict = pickle.load(f)
    
fit = data_dict['fit']
    
if stream == 'atlas_sergey':
    minx, maxx, miny, maxy = -13, 10, -5, 5
    filename = "power_spectra/to_kian.h5"
    with h5py.File(filename, "r") as f:
        hh = f['hh'].value
        hh = hh[20:180]

    myrange = [[minx, maxx], [miny, maxy]]

    xgrid = np.linspace(myrange[0][0], myrange[0][1], hh.shape[0] + 1,
                        True)
    xgrid = xgrid[:-1] + .5 * (xgrid[1] - xgrid[0])
    ygrid = np.linspace(myrange[1][0], myrange[1][1], hh.shape[1] + 1,
                    True)
    ygrid = ygrid[:-1] + .5 * (ygrid[1] - ygrid[0])
    pp1 = xgrid[:, None] + ygrid[None, :] * 0
    pp2 = xgrid[:, None] * 0 + ygrid[None, :]
    mask = ~((pp1>3.5) & (pp1<4.5) & (pp2>-2.8) & (pp2<-1.8))
    vv_mask = hh[mask]
    pp1 = pp1[mask]
    pp2 = pp2[mask]
    
else:
    pp1 = np.load('model_arrays/pp1_atlas.npy')
    pp2 = np.load('model_arrays/pp2_atlas.npy')
    vv_mask = np.load('model_arrays/vv_mask_atlas.npy')
    smooth = np.load('model_arrays/smooth_atlas.npy')

print('Getting model values ...')
xmod = fit['xmod']

xmod_interval = np.apply_along_axis(stats.peak_interval, 0, xmod)
mod_mean_peak = xmod_interval[0,:]
mod_mean_hilo = xmod_interval[1,:]

mod_mean_peak = mod_mean_peak.astype('float64')
model_map_log = np.array(mod_mean_peak).reshape(pp1.shape)
model_map = np.e**model_map_log

print('Getting Stream Intensity means ...')
logint_pix = fit['logint_pix']

logint_pix_interval = np.apply_along_axis(stats.peak_interval, 0, logint_pix)
logint_pix_peak = logint_pix_interval[0,:]
logint_pix_hilo = logint_pix_interval[1,:]

logint_mean = logint_pix_peak.astype('float64')
int_map_log = np.array(logint_mean).reshape(pp1.shape)
int_map = np.e**int_map_log

print('Getting Background Intensity means ...')
logbg_pix = fit['logbg_pix']
logbg_pix_interval = np.apply_along_axis(stats.peak_interval, 0, logbg_pix)
logbg_pix_peak = logbg_pix_interval[0,:]
logbg_pix_hilo = logbg_pix_interval[1,:]


bkg_mean = logbg_pix_peak.astype('float64')
bkg_map_log = np.array(bkg_mean).reshape(pp1.shape)
bkg_map = np.e**bkg_map_log  

print("Getting Width means ...")

width_val = fit['width_val']
widthval_interval = np.apply_along_axis(stats.peak_interval, 0, width_val)
widthval_peak = widthval_interval[0,:]
widthval_hilo = widthval_interval[1,:]


wid_mean = widthval_peak.astype('float64')
width_map = np.array(wid_mean).reshape(pp1.shape)

print("Getting Phi 2 means ...")

fi2_val = fit['fi2_val']
fi2val_interval = np.apply_along_axis(stats.peak_interval, 0, fi2_val)
fi2val_peak = fi2val_interval[0,:]
fi2val_hilo = fi2val_interval[1,:]

phi2_mean = fi2val_peak.astype('float64')
fi2_map = np.array(phi2_mean).reshape(pp1.shape)

print("Getting Log Intensity means ...")

logint_val = fit['logint_val']
logint1d_interval = np.apply_along_axis(stats.peak_interval, 0, logint_val)
logint1d_peak = logint1d_interval[0,:]
logint1d_hilo = logint1d_interval[1,:]

int1d_mean = logint1d_peak.astype('float64')
int1d_map_log = np.array(int1d_mean).reshape(pp1.shape)
int1d_map = np.e**int1d_map_log
test_save = test_extract


np.save('model_arrays/model_map_{}_test{}.npy'.format(stream,test_save), model_map)
np.save('model_arrays/int1d_map_{}_test{}.npy'.format(stream,test_save), int1d_map)
np.save('model_arrays/int_map_{}_test{}.npy'.format(stream,test_save), int_map)
np.save('model_arrays/bkg_map_{}_test{}.npy'.format(stream,test_save), bkg_map)
np.save('model_arrays/width_map_{}_test{}.npy'.format(stream,test_save), width_map)
np.save('model_arrays/fi2_map_{}_test{}.npy'.format(stream,test_save), fi2_map)
np.save('model_arrays/int1d_map_{}_test{}.npy'.format(stream,test_save), int1d_map)
