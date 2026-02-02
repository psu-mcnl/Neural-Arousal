import numpy as np
import xarray as xr
import pandas as pd
from scipy import signal
from scipy import linalg
from scipy import stats
import scipy.io as sio
import os
import shutil
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import pickle
import copy
import random
import seaborn as sns
import warnings

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from tqdm import tqdm

from settings import settings
from utils import *
from filters import filters
from funcs import *

BATCH_SIZE = 2.18689567

cache = settings.neuropixels.cache
session_ob_ids = settings.neuropixels.session_ob_ids
session_fc_ids = settings.neuropixels.session_fc_ids
session_all_ids = settings.neuropixels.session_all_ids

def detect_holes(st, threshold=.007):
    d = np.diff(st)
    ev = d > threshold
    idx = np.where(ev)
    tev = st[idx]
    a = np.diff(tev)
    dur = d[idx]
    return tev, a, dur

def derive_spike_holes(sess_id, resume=False):
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        session_data = cache.get_session_data(sess_id)
        stimTable = session_data.get_stimulus_table()

    units = SessData.units
    probe2units = {}
    for ir, row in units.probe_description.reset_index().iterrows():
        pbid = row.probe_description
        uid = row.unit_id
        if pbid in probe2units.keys():
            probe2units[pbid].append(uid)
        else:
            probe2units[pbid] = [uid]

    hole_info = []
    for pbid in probe2units.keys():
        pb_spike_times = []
        for uid in probe2units[pbid]:
            pb_spike_times.append(session_data.spike_times[uid])
        pb_spike_times = np.concatenate(pb_spike_times)
        pb_spike_times = np.sort(pb_spike_times)
        event_time, intervals, event_dur = detect_holes(pb_spike_times)
        
        d = {'probe_id': pbid,
            'hole_intervals': intervals,
            'hole_duration': event_dur,
            'hole_event_time': event_time}
        hole_info.append(d)
        
    probe_ids = [d['probe_id'] for d in hole_info]

    SpikeHoleData = DataContainer()
    SpikeHoleData.hole_info = hole_info
    SpikeHoleData.probe_ids = probe_ids
    SpikeHoleData.BATCH_SIZE = BATCH_SIZE

    var2save = ['SpikeHoleData']
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    for varName in var2save:
        fName = getattr(settings.projectData.files.sessions, varName)
        fPath = sessDir / fName

        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

def proc_spike_holes(resume=False):
    sessions2proc = session_all_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing session spike holes for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        derive_spike_holes(sess_id, resume=resume)

def main():

    proc_spike_holes(resume=False)

if __name__ == "__main__":
    main()