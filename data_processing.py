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

def process_behavior_data(sess_id, resume=False):

    cfg = DataContainer()
    cfg.tsteps = 0.03
    cfg.stationary_period_thres = 20 # secs
    cfg.edge_effect_thres = 3 # secs
    cfg.running_interval_thres = 0.5 # secs

    """ Re-reference behavior data """
    session_data = cache.get_session_data(sess_id)

    running_df = session_data.running_speed
    running_speed = running_df.velocity.values
    running_time = running_df[['start_time', 'end_time']].mean(axis=1).values

    m = np.isnan(running_time) | np.isnan(running_speed)
    running_time = running_time[~m]
    running_speed = running_speed[~m]

    thr_all = running_speed.std() * 5
    running_speed[running_speed < -thr_all] = 0
    std_neg = np.std(running_speed[running_speed < 0])
    thr_neg = - std_neg * 4.5
    rng = np.random.default_rng(1234)
    noise = rng.normal(0, std_neg, size=(running_speed < thr_neg).sum())
    running_speed[running_speed < thr_neg] = noise

    cfg.thr_all = thr_all
    cfg.thr_neg = thr_neg

    ref_tmin = running_time.min()
    ref_tmax = running_time.max()

    pupil_df = session_data.get_pupil_data()
    if pupil_df is not None:
        pupil_time = pupil_df.index.values
        pupil_size = pupil_df.loc[:, ['pupil_height', 'pupil_width']].mean(axis=1).values
        
        m = np.isnan(pupil_time) | np.isnan(pupil_size)
        pupil_time = pupil_time[~m]
        pupil_size = pupil_size[~m]
        
        ref_tmin = max(ref_tmin, pupil_time.min())
        ref_tmax = min(ref_tmax, pupil_time.max())

    tsteps = cfg.tsteps
    ref_tbins = np.arange(ref_tmin, ref_tmax+tsteps, tsteps)
    ref_time = (ref_tbins[1:] + ref_tbins[:-1]) * 0.5

    rtns = stats.binned_statistic(running_time, running_speed, statistic=np.nanmean, bins=ref_tbins)

    # remove fisrt N-starting or N-ending points for interpolation
    rm_running = np.full(rtns.statistic.shape, False)
    na_idxs = np.isnan(rtns.statistic).nonzero()[0]
    if na_idxs[0] == 0:
        rm_i = na_idxs[(np.diff(na_idxs) > 1).nonzero()[0][0]]
        rm_running[:rm_i+1] = True
    if na_idxs[-1] == rtns.statistic.shape[0] - 1:
        rm_i = na_idxs[(np.diff(na_idxs) > 1).nonzero()[0][-1] + 1]
        rm_running[rm_i:] = True

    # construct interpolation function
    m = np.isnan(rtns.statistic)
    intp_running = interp1d(ref_time[~m], rtns.statistic[~m])

    if pupil_df is not None:
        rtns = stats.binned_statistic(pupil_time, pupil_size, statistic=np.nanmean, bins=ref_tbins)
        
        # remove fisrt N-starting or N-ending points for interpolation
        rm_pupil = np.full(rtns.statistic.shape, False)
        na_idxs = np.isnan(rtns.statistic).nonzero()[0]
        if na_idxs[0] == 0:
            rm_i = na_idxs[(np.diff(na_idxs) > 1).nonzero()[0][0]]
            rm_pupil[:rm_i+1] = True
        if na_idxs[-1] == rtns.statistic.shape[0] - 1:
            rm_i = na_idxs[(np.diff(na_idxs) > 1).nonzero()[0][-1] + 1]
            rm_pupil[rm_i:] = True
        
        # construct interpolation function
        m = np.isnan(rtns.statistic)
        intp_pupil = interp1d(ref_time[~m], rtns.statistic[~m])

    if pupil_df is not None:
        ref_time = ref_time[~(rm_running | rm_pupil)]
    else:
        ref_time = ref_time[~rm_running]
    running_speed = intp_running(ref_time)
    running_speed = xr.DataArray(running_speed, dims=['time'], 
                                coords={'time': ref_time})

    if pupil_df is not None:
        pupil_size = intp_pupil(ref_time)
        pupil_size = xr.DataArray(pupil_size, dims=['time'], 
                                    coords={'time': ref_time})
    else:
        pupil_size = None
        
    running_time = running_speed.time.values
    running_speedf = signal.filtfilt(filters.flt1.b, filters.flt1.a, running_speed.values, method="gust") 
    thres = -np.percentile(running_speedf[running_speedf < 0], .05)
    cfg.thr_run = thres

    running_mask = (running_speedf > thres).astype(int)
    """ stationary mask """
    mdiff = np.diff(1-running_mask, prepend=[0], append=[0])
    mstart_tidx = (mdiff == 1).nonzero()[0]
    mend_tidx = (mdiff == -1).nonzero()[0] - 1
    assert mend_tidx.shape[0] == mstart_tidx.shape[0], 'staionary mask: shape not matched'

    stationary_mask = np.zeros(running_time.shape)
    stationary_period_thres = cfg.stationary_period_thres # secs
    edge_effect_thres = cfg.edge_effect_thres # secs

    for st_idx, ed_idx in zip(mstart_tidx, mend_tidx):
        st_time = running_time[st_idx] + edge_effect_thres
        ed_time = running_time[ed_idx] - edge_effect_thres

        st_idx = np.argmin(np.abs(running_time - st_time))
        ed_idx = np.argmin(np.abs(running_time - ed_time))

        if running_time[ed_idx] - running_time[st_idx] <= stationary_period_thres:
            continue
        stationary_mask[st_idx:(ed_idx+1)] = 1

    """ running mask """
    mdiff = np.diff(running_mask, prepend=[0], append=[0])
    mstart_tidx = (mdiff == 1).nonzero()[0]
    mend_tidx = (mdiff == -1).nonzero()[0] - 1
    assert mend_tidx.shape[0] == mstart_tidx.shape[0], 'running mask: shape not matched'
    
    running_interval_thres = cfg.running_interval_thres # secs
    mstart_tidx_new = [mstart_tidx[0]]
    mend_tidx_new = []

    for st_idx, ed_idx in zip(mstart_tidx[1:], mend_tidx[:-1]):
        st_time = running_time[st_idx]
        ed_time = running_time[ed_idx]
        if st_time - ed_time >= running_interval_thres:
            mstart_tidx_new.append(st_idx)
            mend_tidx_new.append(ed_idx)
    mend_tidx_new.append(mend_tidx[-1])

    running_mask_new = np.zeros(running_time.shape)
    for st_idx, ed_idx in zip(mstart_tidx_new, 
                            mend_tidx_new):
        running_mask_new[st_idx:(ed_idx+1)] = 1

    """ store data """
    running_speedf = xr.DataArray(running_speedf, coords=running_speed.coords)
    running_mask = xr.DataArray(running_mask_new, coords=running_speed.coords)
    stationary_mask = xr.DataArray(stationary_mask, coords=running_speed.coords)

    BehData = DataContainer()

    BehData.running_speed = running_speed
    BehData.running_speedf = running_speedf
    BehData.running_mask = running_mask
    BehData.stationary_mask = stationary_mask

    BehData.pupil_size = pupil_size
    BehData.cfg = cfg

    s = 'All thr: {:.3f}, RUN thr: {:.3f}'.format(cfg.thr_all, cfg.thr_run)
    print(s)

    var2save = ['BehData']
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    for varName in var2save:
        fName = getattr(settings.projectData.files.sessions, varName)
        fPath = sessDir / fName

        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

def extract_session_info(sess_id, resume=False):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        session_data = cache.get_session_data(sess_id)
        stimTable = session_data.get_stimulus_table()

    """ session data """
    SessData = DataContainer()
    SessData.invalid_times = session_data.invalid_times
    SessData.metadata = session_data.metadata
    SessData.probes = session_data.probes
    SessData.unit_structure_acronym = session_data.units.ecephys_structure_acronym
    SessData.units = session_data.units
    SessData.channels = session_data.channels

    SessData.stimulus_epochs = session_data.get_stimulus_epochs()

    var2save = ['SessData']
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    for varName in var2save:
        fName = getattr(settings.projectData.files.sessions, varName)
        fPath = sessDir / fName

        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

def proc_behavior(resume=False):
    sessions2proc = session_all_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing behavior data for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        
        process_behavior_data(sess_id, resume=resume)

def proc_session_info(resume=False):
    sessions2proc = session_all_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing session info data for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        
        extract_session_info(sess_id, resume=resume)

def main():
    # proc_behavior(resume=False)

    proc_session_info(resume=False)

if __name__ == "__main__":
    main()