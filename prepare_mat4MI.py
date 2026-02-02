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
from sklearn.metrics import pairwise_distances
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

def prepare_mat4MI_ob(sess_id, resume=False):

    def save_data_to_file(varData, varName):
        sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)

        fName = getattr(settings.projectData.files.sessions, varName)
        fPath = sessDir / fName
        print('save {:s} to {:s} ...'.format(varName, str(fPath)))
        sio.savemat(fPath, varData)
        print('... done')

    if resume:
        missing = False
        saveDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
        var2save = ['mat4MIcomp']
        for varName in var2save:
            fName = getattr(settings.projectData.files.sessions, varName)
            fPath = saveDir / fName
            if not os.path.isfile(fPath):
                missing = True
                break
        if not missing:
            return

    sessDir = settings.visOrg.dir.sessions / '{:d}'.format(sess_id)
    fDir = sessDir / settings.visOrg.ob_sess_rel_dir.scenes
    varName = 'imgNeuData'
    fName = getattr(settings.visOrg.files.sessions, varName)
    fpath = fDir / fName
    with open(fpath, 'rb') as handle:
        imgNeuData = pickle.load(handle)

    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'CascadeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        CascadeData = pickle.load(handle)
        
    varName = 'SpikeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SpikeData = pickle.load(handle)
        
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)
    
    if not hasattr(CascadeData, 'cascade_posu_onsetInds'):
        mdict = {}
        save_data_to_file(mdict, 'mat4MIcomp')
        return

    spk_abs_time = SpikeData.spike_cnt.time_relative_to_stimulus_onset.values + SpikeData.start_time
    posu_onset_time = spk_abs_time[CascadeData.cascade_posu_onsetInds]

    msk = np.full([posu_onset_time.shape[0]], False)
    for tst, ted in zip([imgNeuData.session_A.tstart, imgNeuData.session_B.tstart,
                        imgNeuData.session_C.tstart], 
                    [imgNeuData.session_A.tstop, imgNeuData.session_B.tstop,
                        imgNeuData.session_C.tstop]):
        msk |= (posu_onset_time > tst) & (posu_onset_time < ted)
    posu_onset_time = posu_onset_time[msk]

    if posu_onset_time.shape[0] == 0:
        mdict = {}
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            session_data = cache.get_session_data(sess_id)
            stimTable = session_data.get_stimulus_table()

        start_time = stimTable.iloc[0].start_time
        stop_time = stimTable.iloc[-1].stop_time
        start_stim_id = stimTable.index[0]

        tstep = 0.05
        tbins = np.arange(start_time, stop_time, tstep) - start_time

        spike_cnt = session_data.presentationwise_spike_counts(tbins, start_stim_id, 
                                                            session_data.units.index.values).squeeze()
        spike_cnt = xr.DataArray(spike_cnt, dims=['time', 'unit_id'], 
                                coords={'time': spike_cnt.time_relative_to_stimulus_onset.values + start_time,
                                        'unit_id': spike_cnt.unit_id.values})
        spike_cnt_z = (spike_cnt - spike_cnt.mean(axis=0)) / spike_cnt.std(axis=0)

        event_SPKidx = np.argmin(pairwise_distances(posu_onset_time.reshape(-1,1), 
                                                    spike_cnt.time.values.reshape(-1,1)), axis=1)
        m = (posu_onset_time - spike_cnt.time.values[event_SPKidx]) < 0.2
        event_SPKidx = event_SPKidx[m]

        wsize = 160
        spk_pupil_segs_z = get_window_signal_around_anchors(spike_cnt_z, 
                            event_SPKidx, winSize=wsize)

        mdict = {}
        for region in SessData.unit_structure_acronym.unique():
            msk = SessData.unit_structure_acronym == region
            if msk.sum() <= 10:
                continue
            sigmean = np.nanmean(spk_pupil_segs_z[:,:, msk], axis=2)
            mdict[region] = sigmean.T
    
    """ save mdict to mat """
    save_data_to_file(mdict, 'mat4MIcomp')

def prepare_mat4MI_fc(sess_id, resume=False):

    def save_data_to_file(varData, varName):
        sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)

        fName = getattr(settings.projectData.files.sessions, varName)
        fPath = sessDir / fName
        print('save {:s} to {:s} ...'.format(varName, str(fPath)))
        sio.savemat(fPath, varData)
        print('... done')

    if resume:
        missing = False
        saveDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
        var2save = ['mat4MIcomp']
        for varName in var2save:
            fName = getattr(settings.projectData.files.sessions, varName)
            fPath = saveDir / fName
            if not os.path.isfile(fPath):
                missing = True
                break
        if not missing:
            return

    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'CascadeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        CascadeData = pickle.load(handle)
        
    varName = 'SpikeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SpikeData = pickle.load(handle)
        
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    if not hasattr(CascadeData, 'cascade_posu_onsetInds'):
        mdict = {}
        save_data_to_file(mdict, 'mat4MIcomp')
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        session_data = cache.get_session_data(sess_id)
        stimTable = session_data.get_stimulus_table()

    spontTable = session_data.get_stimulus_table('spontaneous')
    spontTable = spontTable[spontTable.duration > 120]

    spk_abs_time = SpikeData.spike_cnt.time_relative_to_stimulus_onset.values + SpikeData.start_time
    posu_onset_time = spk_abs_time[CascadeData.cascade_posu_onsetInds]

    msk = np.full([posu_onset_time.shape[0]], False)
    for tst, ted in zip(spontTable.start_time, spontTable.stop_time):
        msk |= (posu_onset_time > tst) & (posu_onset_time < ted)
    posu_onset_time = posu_onset_time[msk]

    if posu_onset_time.shape[0] == 0:
        mdict = {}
    else:
        start_time = stimTable.iloc[0].start_time
        stop_time = stimTable.iloc[-1].stop_time
        start_stim_id = stimTable.index[0]

        tstep = 0.05
        tbins = np.arange(start_time, stop_time, tstep) - start_time

        spike_cnt = session_data.presentationwise_spike_counts(tbins, start_stim_id, 
                                                            session_data.units.index.values).squeeze()
        spike_cnt = xr.DataArray(spike_cnt, dims=['time', 'unit_id'], 
                                coords={'time': spike_cnt.time_relative_to_stimulus_onset.values + start_time,
                                        'unit_id': spike_cnt.unit_id.values})
        spike_cnt_z = (spike_cnt - spike_cnt.mean(axis=0)) / spike_cnt.std(axis=0)

        event_SPKidx = np.argmin(pairwise_distances(posu_onset_time.reshape(-1,1), 
                                                    spike_cnt.time.values.reshape(-1,1)), axis=1)
        m = (posu_onset_time - spike_cnt.time.values[event_SPKidx]) < 0.2
        event_SPKidx = event_SPKidx[m]

        wsize = 160
        spk_pupil_segs_z = get_window_signal_around_anchors(spike_cnt_z, 
                            event_SPKidx, winSize=wsize)

        mdict = {}
        for region in SessData.unit_structure_acronym.unique():
            msk = SessData.unit_structure_acronym == region
            if msk.sum() <= 10:
                continue
            sigmean = np.nanmean(spk_pupil_segs_z[:,:, msk], axis=2)
            mdict[region] = sigmean.T

    """ save mdict to mat """
    save_data_to_file(mdict, 'mat4MIcomp')
    
def proc_ob_sessions(resume=False):
    sessions2proc = session_ob_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'prepare mat file for MI computation for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        try: 
            prepare_mat4MI_ob(sess_id, resume=resume)
        except ValueError:
            print('ERROR: Value Error!')

def proc_fc_sessions(resume=False):
    sessions2proc = session_fc_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'prepare mat file for MI computation for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        try: 
            prepare_mat4MI_fc(sess_id, resume=resume)
        except ValueError:
            print('ERROR: Value Error!')

def proc_sess():
    mdict = {}
    db_names = ['ob', 'fc']
    mdict['ob'] = session_ob_ids
    mdict['fc'] = session_fc_ids

    """ save mdict to mat """
    sessDir = settings.projectData.dir.general
    varName = 'mat_SessEphys'
    fName = getattr(settings.projectData.files.general, varName)
    fPath = sessDir / fName
    print('save {:s} to {:s} ...'.format(varName, str(fPath)))
    sio.savemat(fPath, mdict)
    print('... done')

def main():
    # proc_ob_sessions(resume=False)

    proc_fc_sessions(resume=False)

    # proc_sess()

if __name__ == "__main__":
    main()