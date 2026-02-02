from typing_extensions import dataclass_transform
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from scipy import linalg
from scipy import stats
from scipy.signal import hilbert
import scipy.io as sio
import os
import shutil
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import pickle
import copy
import random
from multiprocessing import Pool, current_process
import seaborn as sns
import warnings
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import lmfit

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import (
    EcephysSession, 
    removed_unused_stimulus_presentation_columns
)
from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot
from allensdk.brain_observatory.visualization import plot_running_speed
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from settings import settings
from utils import *
from filters import filters
from funcs import *

cache = settings.neuropixels.cache
session_ob_ids = settings.neuropixels.session_ob_ids
session_fc_ids = settings.neuropixels.session_fc_ids
session_all_ids = settings.neuropixels.session_all_ids
channels = settings.neuropixels.channels
probes = settings.neuropixels.probes
units = settings.neuropixels.units

def save_state_data(StateData, sess_id):
    var2save = ['StateData']
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    for varName in var2save:
        fName = getattr(settings.projectData.files.sessions, varName)
        fPath = sessDir / fName

        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

def derive_neural_state_index(sess_id, resume=False):

    def tunning_func(x, k=1):
        return 2 / (1 + np.exp(-k * x)) - 1

    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'SpikeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SpikeData = pickle.load(handle)

    varName = 'CascadeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        CascadeData = pickle.load(handle)

    if not hasattr(CascadeData, 'posdly_umsk'):
        print('the subject has no cascade data, skip ...')
        save_state_data(DataContainer(), sess_id)
        return
    
    if (CascadeData.posdly_umsk.sum() < 10) | ((CascadeData.negdly_umsk.sum() < 10)):
        print('the subject has insufficient pos- or neg-delay neurons, skip ...')
        save_state_data(DataContainer(), sess_id)
        return

    nonActive = (SpikeData.spike_cnt.mean(axis=0) < 0.1).values
    CascadeData.posdly_umsk = CascadeData.posdly_umsk & (~nonActive)
    CascadeData.negdly_umsk = CascadeData.negdly_umsk & (~nonActive)
    spk_pos_mean = SpikeData.spike_cnt_n[:, CascadeData.posdly_umsk].mean(axis=1)
    spk_neg_mean = SpikeData.spike_cnt_n[:, CascadeData.negdly_umsk].mean(axis=1)

    std_pos = spk_pos_mean[SpikeData.stationary_mask].std()
    std_neg = spk_neg_mean[SpikeData.stationary_mask].std()
    w_pos = 2 * std_neg / (std_neg + std_pos)
    w_neg = 2 * std_pos / (std_neg + std_pos)

    state_index = np.zeros(spk_pos_mean.shape[0])
    for k, (spk_pos, spk_neg) in enumerate(zip(spk_pos_mean[1:], spk_neg_mean[1:])):
        delta = w_pos * spk_pos - w_neg * spk_neg
        state_index[k] = tunning_func(state_index[k-1] + delta, k=1)

    state_index = xr.DataArray(state_index, coords=spk_pos_mean.coords)

    StateData = DataContainer()
    StateData.state_index = state_index

    """ save data """
    save_state_data(StateData, sess_id)

def derive_state_threshold(sess_id, resume=False):
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'SpikeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SpikeData = pickle.load(handle)

    varName = 'CascadeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        CascadeData = pickle.load(handle)
        
    varName = 'StateData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        StateData = pickle.load(handle)
        
    varName = 'BehData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        BehData = pickle.load(handle)
        
    if not hasattr(StateData, 'state_index'):
        save_state_data(DataContainer(), sess_id)
        return
        
    # derive threshold by fitting the data with Gaussian mixture model
    X = StateData.state_index[SpikeData.stationary_mask].values
    X = X[~np.isnan(X)]
    gm = BayesianGaussianMixture(n_components=2, random_state=0, n_init=3,
                                init_params='k-means++', covariance_type='full', max_iter=200)
    gm.fit(X.reshape(-1,1))
    x_base = np.arange(-1, 1, 0.001)
    pdf1 = gm.weights_[0] * stats.norm.pdf(x_base, gm.means_[0],
                              np.sqrt(gm.covariances_[0].item()))
    pdf2 = gm.weights_[1] * stats.norm.pdf(x_base, gm.means_[1],
                                np.sqrt(gm.covariances_[1].item()))    
    thr_ind = signal.find_peaks(-(pdf1 - pdf2) ** 2)[0]
    thr_ind = thr_ind[np.argmax(pdf1[thr_ind])]
    StateData.state_thr_from_GM = x_base[thr_ind]

    s = '[Gaussian Mixture] state thr = {:.3f}\n'.format(StateData.state_thr_from_GM)
    print(s)

    """ save data """
    save_state_data(StateData, sess_id)

def proc_derive_neural_state_index(resume=False):
    sessions2proc = session_all_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'NEURAL STATE INDEX'
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        s += ' for {:d} [{:d}|{:d}]'.format(sess_id, i+1, len(sessions2proc))
        print(s)
        derive_neural_state_index(sess_id, resume=resume)

def proc_derive_state_threshold(resume=False):
    sessions2proc = session_all_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'NEURAL STATE INDEX THRESHOLD'
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        s += ' for {:d} [{:d}|{:d}]'.format(sess_id, i+1, len(sessions2proc))
        print(s)
        derive_state_threshold(sess_id, resume=resume)

def main():
    proc_derive_neural_state_index(resume=False)

    proc_derive_state_threshold(resume=False)

if __name__ == "__main__":
    main()