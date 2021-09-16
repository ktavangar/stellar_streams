import numpy as np
import pickle
import matplotlib.pyplot as plt
#import pystan.experimental

print('open model')
with open("phoenix_models/phoenix_model_fit_test15.pickle", "rb") as f:
    data_dict = pickle.load(f)

fit = data_dict['fit']

parnames = fit.flatnames

print('Getting xmvals ...')
xmvals = []
for i in parnames:
    if 'xmod' in i:
        xmvals = np.append(xmvals, i)

print('Getting means ...')
mod_mean = []
for i in xmvals:
    mod_mean = np.append(mod_mean, fit[i].mean())

pp1 = np.load('pp1_phoenix.npy')
pp2 = np.load('pp2_phoenix.npy')

model_map_log = np.array(mod_mean).reshape(pp1.shape)
model_map = np.e**model_map_log
np.save('model_arrays/model_map_test17.npy', model_map)

intvals = []
for i in parnames:
    if 'logint_pix' in i:
        intvals = np.append(intvals, i)

print('Getting Stream Intensity means ...')
int_mean = []
for i in intvals:
    int_mean = np.append(int_mean, fit[i].mean())

int_map_log = np.array(int_mean).reshape(pp1.shape)
int_map = np.e**int_map_log

bkgvals = []
for i in parnames:
    if 'logbg_pix' in i:
        bkgvals = np.append(bkgvals, i)

print('Getting background means ...')
bkg_mean = []
for i in bkgvals:
    bkg_mean = np.append(bkg_mean, fit[i].mean())

bkg_map_log = np.array(bkg_mean).reshape(pp1.shape)
bkg_map = np.e**bkg_map_log  

print("Getting Width means ...")
w = []
for i in parnames:
    if ('width_val' in i) & ('log' not in i):
        w = np.append(w,i)

wid_mean = []
wid_high = []
wid_low = []

for i in w:
    ave = np.mean(fit[i])
    wid_mean = np.append(wid_mean, ave)
    std = np.std(fit[i])
    wid_high = np.append(wid_high, ave+std)
    wid_low = np.append(wid_low, ave-std)

width_map = np.array(wid_mean).reshape(pp1.shape)
width_high_map = np.array(wid_high).reshape(pp1.shape)
width_low_map = np.array(wid_low).reshape(pp1.shape)

print("Getting Phi 2 means ...")
phi2 = []
for i in parnames:
    if 'fi2_val' in i:
        phi2 = np.append(phi2, i)

phi2_mean = []
phi2_high = []
phi2_low = []

for i in phi2:
    ave= np.mean(fit[i])
    phi2_mean = np.append(phi2_mean, ave)
    std = np.std(fit[i])
    phi2_high = np.append(phi2_high, ave+std)
    phi2_low = np.append(phi2_low, ave-std)

fi2_map = np.array(phi2_mean).reshape(pp1.shape)
fi2_high_map = np.array(phi2_high).reshape(pp1.shape)
fi2_low_map = np.array(phi2_low).reshape(pp1.shape)

print("Getting Log Intensity means ...")
int1d = []
for i in parnames:
    if 'logint_val' in i:
        int1d = np.append(int1d, i)

int1d_mean = []
int1d_high = []
int1d_low = []

for i in int1d:
    ave= np.mean(fit[i])
    int1d_mean = np.append(int1d_mean, ave)
    std = np.std(fit[i])
    int1d_high = np.append(int1d_high, ave+std)
    int1d_low = np.append(int1d_low, ave-std)

int1d_map_log = np.array(int1d_mean).reshape(pp1.shape)
int1d_high_map_log = np.array(int1d_high).reshape(pp1.shape)
int1d_low_map_log = np.array(int1d_low).reshape(pp1.shape)

int1d_map = np.e**int1d_map_log
int1d_high_map = np.e**int1d_high_map_log
int1d_low_map = np.e**int1d_low_map_log
np.save('model_arrays/int1d_map_test17.npy', int1d_map)

vv_mask = np.load('vv_mask_phoenix.npy')
smooth = np.load('smooth_phoenix.npy')

print("Plotting...")
fig, axs = plt.subplots(4,2, figsize=(15,5))

axs[0][0].pcolormesh(pp1, pp2, smooth, vmin = 4, vmax = 9, cmap='gray_r')
axs[0][0]
.set_xlim(7.5, -9)
axs[0][0].set_xlabel('Phi 1')
axs[0][0].set_title('Phoenix Stream Data')

axs[1][0].pcolormesh(pp1, pp2, model_map, vmin = 4, vmax = 9, cmap='gray_r')
axs[1][0].set_xlim(7.5, -9)
axs[1][0].set_xlabel('Phi 1')
axs[1][0].set_title('Phoenix Stream Model')

axs[2][0].pcolormesh(pp1, pp2, int_map_log, vmin = 0, vmax =5, cmap='gray_r')
axs[2][0].set_xlim(7.5, -9)
axs[2][0].set_xlabel('Phi 1')
axs[2][0].set_title('Stream Model w/o Background')

axs[3][0].pcolormesh(pp1, pp2, bkg_map, vmin = 4,vmax = 9, cmap='gray_r')
axs[3][0].set_xlim(7.5, -9)
axs[3][0].set_xlabel('Phi 1')
axs[3][0].set_title('Background Model')

axs[0][1].pcolormesh(pp1, pp2, vv_mask-model_map, vmin = -1.5,vmax = 1.5, cmap='gray_r')
axs[0][1].set_xlim(7.5, -9)
axs[0][1].set_xlabel('Phi 1')
axs[0][1].set_title('Residual')

axs[1][1].plot(pp1[0], int1d_map[0])
axs[1][1].set_xlim(7.5, -9)
axs[1][1].set_title('Stream Intensity')

axs[2][1].plot(pp1[0], width_map[0])
axs[2][1].set_xlim(7.5, -9)
axs[2][1].set_title('Stream Width')

axs[3][1].plot(pp1[0], fi2_map[0])
axs[3][1].set_xlim(7.5, -9)
axs[3][1].set_title('Stream Track')
plt.show()
