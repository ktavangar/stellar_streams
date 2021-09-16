import numpy as np
import pickle
import matplotlib.pyplot as plt
import ugali
from ugali.utils import stats

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--test', default=7)
    parser.add_argument('-d', '--data', default='all')
    args = parser.parse_args()

    test = args.test
    data = args.data
    
    hh = np.load('phoenix_data_catalogs/phoenix_stream_coord_catalog_paper_final.npy')
    hh = np.transpose(hh)
    pp1 = np.load('model_arrays/pp1_full_phoenix_tall.npy')
    pp1 = pp1[:,1:-1]
    pp2 = np.load('model_arrays/pp2_full_phoenix_tall.npy') 
    pp2 = pp2[:,1:-1]
    hh[np.where((hh==0) & (pp1<-6))] = np.nan
    hh[np.where(hh>16)] = np.nan
    hh = hh.astype('int')
    vv_mask=hh
    mask = hh > -1

    smooth_full = np.load('model_arrays/smooth_full_phoenix_tall.npy')[:,1:]

    print('open model')
    with open("phoenix_models/phoenix_model_fit_final_test{}.pickle".format(str(test)), "rb") as f:
        data_dict = pickle.load(f)

    fit = data_dict['fit']
    parnames = fit.flatnames

    print("Getting Log Intensity means ...")

    logint_val = fit['logint_val']
    logint1d_peak = np.zeros(pp1.shape)
    logint1d_std_lo = np.zeros(pp1.shape)
    logint1d_std_hi = np.zeros(pp1.shape)

    logint1d_interval = np.apply_along_axis(stats.peak_interval, 0, logint_val)

    logint1d_lo = []
    logint1d_hi = []
    for i in range(len(logint1d_interval[1,:])):
        logint1d_lo = np.append(logint1d_lo, np.asarray(logint1d_interval[1,:][i])[0])
        logint1d_hi = np.append(logint1d_hi, np.asarray(logint1d_interval[1,:][i])[1])
        if logint1d_interval[0][i] > np.asarray(logint1d_interval[1,:][i])[1]:
            logint1d_interval[0][i] = np.log(0.5*(np.e**np.asarray(logint1d_interval[1,:][i])[0] + 
                                              np.e**np.asarray(logint1d_interval[1,:][i])[1]))
    #see if i can fix this in a better way for the final version
    logint1d_std_lo[mask] = logint1d_lo
    logint1d_std_hi[mask] = logint1d_hi
    logint1d_peak[mask] = logint1d_interval[0,:]

    int1d_mean = logint1d_peak.astype('float64')
    int1d_map_log = np.array(int1d_mean).reshape(pp1.shape)
    int1d_map = np.e**int1d_map_log
    logint1d_hi_map = logint1d_std_hi
    logint1d_lo_map = logint1d_std_lo
    int1d_hi_map = np.e**logint1d_hi_map
    int1d_lo_map = np.e**logint1d_lo_map

    np.save('model_arrays/phoenix_int1d_map_final_test{}.npy'.format(test), int1d_map)
    np.save('model_arrays/phoenix_int1d_hi_map_final_test{}.npy'.format(test), int1d_hi_map)
    np.save('model_arrays/phoenix_int1d_lo_map_final_test{}.npy'.format(test), int1d_lo_map)

    if data == 'all':
    
        print('Getting Background Intensity means ...')
        logbg_pix = fit['logbg_pix']
        logbg_pix_peak = np.zeros(pp1.shape)

        logbg_pix_interval = np.apply_along_axis(stats.peak_interval, 0, logbg_pix)

        logbg_pix_peak[mask] = logbg_pix_interval[0,:]
        logbg_mean = logbg_pix_peak.astype('float64')
        bg_map_log = np.array(logbg_mean).reshape(pp1.shape)
        bg_map = np.e**bg_map_log

        
        print("Getting Width means ...")
        width_val = fit['width_val']
        widthval_peak = np.zeros(pp1.shape)
        widthval_std_lo = np.zeros(pp1.shape)
        widthval_std_hi = np.zeros(pp1.shape)

        widthval_interval = np.apply_along_axis(stats.peak_interval, 0, width_val)

        widthval_peak[mask] = widthval_interval[0,:]
        widthval_lo = []
        widthval_hi = []
        for i in range(len(widthval_interval[1,:])):
            widthval_lo = np.append(widthval_lo, np.asarray(widthval_interval[1,:][i])[0])
            widthval_hi = np.append(widthval_hi, np.asarray(widthval_interval[1,:][i])[1])
        widthval_std_lo[mask] = widthval_lo
        widthval_std_hi[mask] = widthval_hi

        wid_mean = widthval_peak.astype('float64')
        width_map = np.array(wid_mean).reshape(pp1.shape)
        width_hi_map = widthval_std_hi
        width_lo_map = widthval_std_lo

       
        print("Getting Phi 2 means ...")
        fi2_val = fit['fi2_val']
        fi2val_peak = np.zeros(pp1.shape)
        fi2val_std_lo = np.zeros(pp1.shape)
        fi2val_std_hi = np.zeros(pp1.shape)

        fi2val_interval = np.apply_along_axis(stats.peak_interval, 0, fi2_val)

        fi2val_peak[mask] = fi2val_interval[0,:]
        fi2val_lo = []
        fi2val_hi = []
        for i in range(len(fi2val_interval[1,:])):
            fi2val_lo = np.append(fi2val_lo, np.asarray(fi2val_interval[1,:][i])[0])
            fi2val_hi = np.append(fi2val_hi, np.asarray(fi2val_interval[1,:][i])[1])
        fi2val_std_lo[mask] = fi2val_lo
        fi2val_std_hi[mask] = fi2val_hi

        phi2_mean = fi2val_peak.astype('float64')
        fi2_map = np.array(phi2_mean).reshape(pp1.shape)
        fi2_hi_map = fi2val_std_hi
        fi2_lo_map = fi2val_std_lo
        
        
        print('Saving arrays ...')
        np.save('model_arrays/phoenix_bkg_map_final_test{}.npy'.format(test), bg_map)
        np.save('model_arrays/phoenix_width_map_final_test{}.npy'.format(test), width_map)
        np.save('model_arrays/phoenix_width_hi_map_final_test{}.npy'.format(test), width_hi_map)
        np.save('model_arrays/phoenix_width_lo_map_final_test{}.npy'.format(test), width_lo_map)
        np.save('model_arrays/phoenix_fi2_map_final_test{}.npy'.format(test), fi2_map)
        np.save('model_arrays/phoenix_fi2_hi_map_final_test{}.npy'.format(test), fi2_hi_map)
        np.save('model_arrays/phoenix_fi2_lo_map_final_test{}.npy'.format(test), fi2_lo_map)
        
        print('Done')
