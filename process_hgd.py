import numpy as np
import os
import logging
from collections import OrderedDict
from braindecode.datautil.signalproc import exponential_running_standardize, highpass_cnt
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from scipy import signal
from braindecode.datasets.bbci import BBCIDataset
from scipy.io import savemat
import config
from main import build_datasets_files

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

"""
    # This code requires an older version of Braindecode to function properly.
    # To install the compatible version, use:
    # pip install https://github.com/TNTLFreiburg/braindecode/archive/master.zip
"""
    

def preprocess_hgd(data_file, low_cut_hz, save_path, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    loader = BBCIDataset(data_file, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def, clean_ival)
    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    data = dataset.X[clean_trial_mask]
    labels = dataset.y[clean_trial_mask]
    savemat(os.path.join(save_path, 'evaluation.mat'), {'data': dataset.X, 'labels': dataset.y})

    return dataset

if __name__ == '__main__':
    base_save_path = r'E:\EEG\dataset'
    #train_datasets = build_datasets_files(stage='train')
    test_datasets = build_datasets_files(stage='test')
    for i in range(len(test_datasets)):
        subject = test_datasets[i][0].split('/')[-2]
        print(f'------start {subject} processing------')
        save_path = os.path.join(base_save_path, "NewHGD", subject)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #train_file = train_datasets[i]
        test_file = test_datasets[i]
        #filename = train_file[0]
        filename = test_file[0]
        preprocess_hgd(filename, 4, save_path)
