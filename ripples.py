import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from scipy import linalg
from scipy import stats
import scipy.io as sio
import os
import shutil
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy import interpolate
import pickle
import copy
import random
from tqdm import tqdm
import seaborn as sns
import warnings

import cv2
from torchinfo import summary
import pytorch_lightning as pl

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import pairwise_distances

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import (
    EcephysSession, 
    removed_unused_stimulus_presentation_columns
)

from settings import settings
from utils import *
from filters import filters
from funcs import *


def ripple_detection(sig, sr):
    """ return ripple event mask and ripple peak index 
        input:
            sig: lfp data in the form of N time points and M channels
            sr: sampling rate of the lfp signal
    """
    lowB = 80
    highB = 250
    SDthr = 5
    b1, a1 = signal.butter(2, Wn = [lowB/sr*2, highB/sr*2], btype = "bandpass")
    b2, a2 = signal.butter(2, Wn = (lowB+highB)/2/3.1415926/sr*2, btype = "lowpass")
    ripplemsk = []
    ripplectrind = []
    sig = signal.filtfilt(b1, a1, sig, axis=0,method="gust") 
    #for cnt in range(sig.shape[1]):
    for cnt in tqdm(range(sig.shape[1])):
    
        org = sig[:,cnt]
        clipped = np.clip(org,org.mean()-org.std()*SDthr,org.mean()+org.std()*SDthr)
        clipped_pw = signal.filtfilt(b2, a2, abs(clipped),axis=0,method='gust')
        org_pw = signal.filtfilt(b2, a2, abs(org),axis=0,method='gust')

        sd5msk = (org_pw>(clipped_pw.mean()+clipped_pw.std()*SDthr)).astype(int)
        sd5msk[[0,-1]]=0
        # remove small ripple mask (<15ms)
        sd5msk_diff = np.diff(sd5msk)
        sd5startind = np.asarray(np.where(sd5msk_diff==1)).squeeze()+1
        sd5endind = np.asarray(np.where(sd5msk_diff==-1)).squeeze()+1
        sd5blksz = sd5endind-sd5startind
        for cnt2 in range(sd5blksz.shape[0]):
            if sd5blksz[cnt2] < 15/1000*sr:
                sd5msk[sd5startind[cnt2]:sd5endind[cnt2]]=0
        # merge ripple mask with a small gap (<15ms)
        sd5msk_diff = np.diff(sd5msk)
        sd5startind = np.asarray(np.where(sd5msk_diff==1)).squeeze()+1
        sd5endind = np.asarray(np.where(sd5msk_diff==-1)).squeeze()+1
        if sd5startind.size>1:
            sd5gapsz = sd5startind[1:]-sd5endind[:-1]
            for cnt3 in range(sd5gapsz.size):
                if sd5gapsz[cnt3] < 15/1000*sr:
                    sd5msk[sd5endind[cnt3]:sd5startind[cnt3+1]]=1
        else:
            sd5msk = np.zeros(sd5msk.shape)
            
        ripplepwpksind, _ = signal.find_peaks(sd5msk*org_pw)
        
        sd2msk = (org_pw>(clipped_pw.mean()+clipped_pw.std()*2)).astype(int)
        sd2msk[[0,-1]]=0
        sd2msk_diff = np.diff(sd2msk)
        sd2startind = np.asarray(np.where(sd2msk_diff==1)).squeeze()+1
        sd2endind = np.asarray(np.where(sd2msk_diff==-1)).squeeze()+1
        for cnt4 in range(sd2startind.size):
            if np.sum((ripplepwpksind-sd2startind[cnt4])*(ripplepwpksind-sd2endind[cnt4])<0)==0:
                sd2msk[sd2startind[cnt4]:sd2endind[cnt4]]=0
        # print('processed channel [{:d}|{:d}]'.format(cnt+1, sig.shape[1]))
        ripplemsk.append(sd2msk*org_pw)
        
        if ripplepwpksind.size==0:
            ripplectrind.append(np.nan)
        else:
            hfnpks, _ = signal.find_peaks(-org)
            ripplectrind.append(hfnpks[np.abs(np.subtract.outer(hfnpks, ripplepwpksind)).argmin(0)])
            
    return ripplemsk, ripplectrind

cache = settings.neuropixels.cache
session_ob_ids = settings.neuropixels.session_ob_ids
session_fc_ids = settings.neuropixels.session_fc_ids
session_all_ids = settings.neuropixels.session_all_ids
channels = settings.neuropixels.channels
probes = settings.neuropixels.probes
units = settings.neuropixels.units

def natural_scene_image_ripples(sess_id, resume=False):

    def save_data_to_file(RippleData):
        var2save = ['RippleData']
        sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
        for varName in var2save:
            fName = getattr(settings.projectData.files.sessions, varName)
            fPath = sessDir / fName

            with open(fPath, 'wb') as handle:
                print('save {:s} to {:s} ...'.format(varName, str(fPath)))
                pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('... done')

    if resume:
        missing = False
        saveDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
        var2save = ['RippleData']
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
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    CA1_channelIDs = SessData.channels[SessData.channels.ecephys_structure_acronym == 'CA1'].index.values
    RippleData = DataContainer()
    if len(CA1_channelIDs) == 0:
        save_data_to_file(RippleData)
        print('No CA1 channel found, exit ...')
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        session_data = cache.get_session_data(sess_id)
        _ = session_data.get_stimulus_table()

    probes_id = session_data.probes.index.values

    """ load lfp data """
    lfps = list()
    for i, pb_id in enumerate(probes_id):
        print('processing probe {:d} [{:d}|{:d}]'.format(pb_id, i+1, len(probes_id)))
        try:
            lfp = session_data.get_lfp(pb_id)
            lfps.append(lfp[:,lfp.channel.isin(CA1_channelIDs)])
        except ValueError:
            print('lfp data not found')

    num_CA1Channels = sum([lfp.channel.shape[0] for lfp in lfps])
    if num_CA1Channels == 0:
        save_data_to_file(RippleData)
        print('No CA1 channel found, exit ...')
        return

    for sessName in ['session_A', 'session_B', 'session_C']:
        sessRplData = DataContainer()
        setattr(RippleData, sessName, sessRplData)

    """ get session lfp data """
    for sessName in ['session_A', 'session_B', 'session_C']:
        sessRplData = getattr(RippleData, sessName)
        
        tstart = getattr(imgNeuData, sessName).tstart
        tstop = getattr(imgNeuData, sessName).tstop
        tpad = 2 #s
        sr = 1250
        tstart -= tpad
        tstop += tpad
        
        ref_time = np.arange(tstart, tstop, 1 / sr)
        ref_time = ref_time[ref_time > tstart]
        ref_time = ref_time[ref_time < tstop]
        lfps_sess = [re_reference_lfps(lfp, ref_time) for lfp in lfps]
        lfps_sess = xr.concat(lfps_sess, dim='channel')
        lfps_sess = lfps_sess.fillna(0)
        
        sessRplData.lfps_CA1 = lfps_sess
        sessRplData.sr = sr
        sessRplData.tpad = tpad

    """ compute ripples """
    for sessName in ['session_A', 'session_B', 'session_C']:
        print('computing ripple events for natural scene imagee {:s}'.format(sessName))
        sessRplData = getattr(RippleData, sessName)
        
        ripplemsk, ripplectrind = ripple_detection(sessRplData.lfps_CA1.values.copy(), 
                                                sessRplData.sr)
        ripplemsk = np.array(ripplemsk).T
        ripplemsk = xr.DataArray(ripplemsk, coords=sessRplData.lfps_CA1.coords)
        
        
        sessRplData.rippleMsk = ripplemsk
        sessRplData.rippleCtrInd = ripplectrind
        delattr(sessRplData, 'lfps_CA1')

    save_data_to_file(RippleData)

def proc_ripples(resume=False):
    sessions2proc = session_ob_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing ripple data for image tasks for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        try: 
            natural_scene_image_ripples(sess_id, resume=resume)
        except ValueError:
            print('ERROR: Value Error!')

def main():

    proc_ripples(resume=True)


if __name__ == "__main__":
    main()
    

        