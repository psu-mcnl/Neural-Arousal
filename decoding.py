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

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

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

mcc = MouseConnectivityCache(resolution = 25, 
                             manifest_file = settings.connectivity.manifest_path)
structure_tree = mcc.get_structure_tree()

name_map = structure_tree.get_name_map() # dictionary mapping ids to structure names
acrnm_map = structure_tree.get_id_acronym_map() # dictionary mapping acronyms to ids
colormap = structure_tree.get_colormap() # the colormap used for the allen 3D mouse atlas ontology
id_map = structure_tree.value_map(lambda x: x['id'], lambda y: y['acronym'])

def conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks):
    kfTrain_msks = [[] for _ in range(len(cfg.stim_clsIds))]
    kfTest_msks = [[] for _ in range(len(cfg.stim_clsIds))]

    kf = KFold(n_splits=cfg.k, shuffle=True, random_state=0)

    for ci in range(len(cfg.stim_clsIds)):
        valid = cond_msks[ci]
        if sum(valid) < cfg.min_sample_per_fold:
            train_msk = np.full([cfg.k, valid.shape[0]], False)
            test_msk = np.full([cfg.k, valid.shape[0]], False)
            kfTrain_msks[ci] = train_msk
            kfTest_msks[ci] = test_msk
            continue
        
        valid_repeats = valid.nonzero()[0]
        for train_idx, test_idx in kf.split(valid_repeats):
            train_msk = np.full([valid.shape[0]], False)
            test_msk = np.full([valid.shape[0]], False)
            train_msk[valid_repeats[train_idx]] = True
            test_msk[valid_repeats[test_idx]] = True

            kfTrain_msks[ci].append(train_msk)
            kfTest_msks[ci].append(test_msk)

    kfTrain_msks = np.array(kfTrain_msks).transpose((1,0,2))
    kfTest_msks = np.array(kfTest_msks).transpose((1,0,2))
    
    return kfTrain_msks, kfTest_msks

def train_test_kFold(DecoderData, NeuEsmbl, RspData, cfg):

    for esmbl_name in NeuEsmbl.get_registered_esmbls():
        esmbl = getattr(NeuEsmbl, esmbl_name)
        dec = DataContainer()
        setattr(DecoderData, esmbl_name, dec)

        print('* processing {:s}'.format(esmbl.name))
        if esmbl.umsk.sum() < 5:
            print('--insufficient units, skipped')
            dec.hit = None
            dec.intervals = None
            dec.stationary = None
            dec.running = None
            dec.high_state = None
            dec.low_state = None
            dec.sample_idxs = None
            dec.mdl_params = None
            continue

        test_hit = []
        test_intervals = []
        test_sample_idxs = []
        test_stationary = []
        test_running = []
        test_Hstate = []
        test_Lstate = []
        mdl_params = []

        log = {'train_acc': [], 'test_acc': [],
               'train_cls': [], 'test_cls': [],
               'train_smpl': [], 'test_smpl': []}

        for k in range(cfg.k):
            train_msk = DecoderData.kfTrain_msks[k]
            test_msk = DecoderData.kfTest_msks[k]

            if train_msk.any(axis=1).sum() < cfg.min_classes_per_fold:
                continue
                
            feat_train = RspData.spike_cnt_n.values[train_msk.T][:,esmbl.umsk]
            feat_test = RspData.spike_cnt_n.values[test_msk.T][:,esmbl.umsk]

            tgt_train = RspData.targets.T[train_msk.T]
            tgt_test = RspData.targets.T[test_msk.T]

            scaler = StandardScaler()
            scaler.fit(feat_train)
            feat_train = scaler.transform(feat_train)
            feat_test = scaler.transform(feat_test)

            if cfg.classifier == 'SVM':
                clf = SVC(kernel='linear', C=1, random_state=0)
            
            if cfg.classifier == 'LR':
                clf = LogisticRegression(random_state=0)
                
            if cfg.classifier == 'NN':
                clf = MLPClassifier(hidden_layer_sizes=(512,), random_state=0, alpha=0.025, activation='tanh')
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                clf.fit(feat_train, tgt_train)
            
            tgt_preds = clf.predict(feat_test)
            
            # variables to save
            test_hit.append((tgt_test == tgt_preds).astype(int))
            test_intervals.append(RspData.intervals.transpose([1,0,2])[test_msk.T])
            test_stationary.append(RspData.stationary.T[test_msk.T])
            test_running.append(RspData.running.T[test_msk.T])
            test_Hstate.append(RspData.high_state_msk.T[test_msk.T])
            test_Lstate.append(RspData.low_state_msk.T[test_msk.T])
            test_sample_idxs.append(RspData.sample_idxs.T[test_msk.T])
            if cfg.classifier == 'SVM':
                mdl_params.append(clf.coef_)

            log['train_acc'].append(accuracy_score(tgt_train, clf.predict(feat_train)))
            log['test_acc'].append(accuracy_score(tgt_test, tgt_preds))
            log['train_cls'].append(np.unique(tgt_train).shape[0])
            log['test_cls'].append(np.unique(tgt_test).shape[0])
            log['train_smpl'].append(tgt_train.shape[0])
            log['test_smpl'].append(tgt_test.shape[0])
        
        if len(test_hit) > 0:
            dec.hit = np.concatenate(test_hit, axis=0)
            dec.intervals = np.concatenate(test_intervals, axis=0)
            dec.stationary = np.concatenate(test_stationary, axis=0)
            dec.running = np.concatenate(test_running, axis=0)
            dec.high_state = np.concatenate(test_Hstate, axis=0)
            dec.low_state = np.concatenate(test_Lstate, axis=0)
            dec.sample_idxs = np.concatenate(test_sample_idxs, axis=0)
            dec.mdl_params = mdl_params

            print('--Accuracy: [training: {:.3f}, testing: {:.3f}]'.format(
                    np.mean(log['train_acc']),
                    np.mean(log['test_acc'])))
            print('--Classes: [training: {:s}, testing: {:s}]'.format(
                    ', '.join(['{:d}'.format(n) for n in log['train_cls']]),
                    ', '.join(['{:d}'.format(n) for n in log['test_cls']])))
            print('--Samples: [training: {:s}, testing: {:s}]'.format(
                    ', '.join(['{:d}'.format(n) for n in log['train_smpl']]),
                    ', '.join(['{:d}'.format(n) for n in log['test_smpl']])))
        else:
            dec.hit = None
            dec.intervals = None
            dec.stationary = None
            dec.running = None
            dec.high_state = None
            dec.low_state = None
            dec.sample_idxs = None
            dec.mdl_params = None

def train_test_kFold_exclude_regions(DecoderData, NeuEsmbl, RspData, cfg):

    for esmbl_name in NeuEsmbl.get_registered_esmbls():
        esmbl = getattr(NeuEsmbl, esmbl_name)
        dec = DataContainer()
        setattr(DecoderData, esmbl_name, dec)

        print('* processing {:s} (excluding region)'.format(esmbl.name))
        if (esmbl.umsk.sum() < 40) or ((~esmbl.umsk).sum() < 5):
            print('--insufficient units, skipped')
            dec.hit = None
            dec.intervals = None
            dec.stationary = None
            dec.running = None
            dec.high_state = None
            dec.low_state = None
            dec.sample_idxs = None
            dec.mdl_params = None
            continue

        test_hit = []
        test_intervals = []
        test_sample_idxs = []
        test_stationary = []
        test_running = []
        test_Hstate = []
        test_Lstate = []
        mdl_params = []

        log = {'train_acc': [], 'test_acc': [],
               'train_cls': [], 'test_cls': [],
               'train_smpl': [], 'test_smpl': []}

        for k in range(cfg.k):
            train_msk = DecoderData.kfTrain_msks[k]
            test_msk = DecoderData.kfTest_msks[k]

            if train_msk.any(axis=1).sum() < cfg.min_classes_per_fold:
                continue
                
            feat_train = RspData.spike_cnt_n.values[train_msk.T][:,~esmbl.umsk]
            feat_test = RspData.spike_cnt_n.values[test_msk.T][:,~esmbl.umsk]

            tgt_train = RspData.targets.T[train_msk.T]
            tgt_test = RspData.targets.T[test_msk.T]

            scaler = StandardScaler()
            scaler.fit(feat_train)
            feat_train = scaler.transform(feat_train)
            feat_test = scaler.transform(feat_test)

            if cfg.classifier == 'SVM':
                clf = SVC(kernel='linear', C=1, random_state=0)
            
            if cfg.classifier == 'LR':
                clf = LogisticRegression(random_state=0)
                
            if cfg.classifier == 'NN':
                clf = MLPClassifier(hidden_layer_sizes=(512,), random_state=0, alpha=0.025, activation='tanh')
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                clf.fit(feat_train, tgt_train)
            
            tgt_preds = clf.predict(feat_test)
            
            # variables to save
            test_hit.append((tgt_test == tgt_preds).astype(int))
            test_intervals.append(RspData.intervals.transpose([1,0,2])[test_msk.T])
            test_stationary.append(RspData.stationary.T[test_msk.T])
            test_running.append(RspData.running.T[test_msk.T])
            test_Hstate.append(RspData.high_state_msk.T[test_msk.T])
            test_Lstate.append(RspData.low_state_msk.T[test_msk.T])
            test_sample_idxs.append(RspData.sample_idxs.T[test_msk.T])
            if cfg.classifier == 'SVM':
                mdl_params.append(clf.coef_)

            log['train_acc'].append(accuracy_score(tgt_train, clf.predict(feat_train)))
            log['test_acc'].append(accuracy_score(tgt_test, tgt_preds))
            log['train_cls'].append(np.unique(tgt_train).shape[0])
            log['test_cls'].append(np.unique(tgt_test).shape[0])
            log['train_smpl'].append(tgt_train.shape[0])
            log['test_smpl'].append(tgt_test.shape[0])
        
        if len(test_hit) > 0:
            dec.hit = np.concatenate(test_hit, axis=0)
            dec.intervals = np.concatenate(test_intervals, axis=0)
            dec.stationary = np.concatenate(test_stationary, axis=0)
            dec.running = np.concatenate(test_running, axis=0)
            dec.high_state = np.concatenate(test_Hstate, axis=0)
            dec.low_state = np.concatenate(test_Lstate, axis=0)
            dec.sample_idxs = np.concatenate(test_sample_idxs, axis=0)
            dec.mdl_params = mdl_params

            print('--Accuracy: [training: {:.3f}, testing: {:.3f}]'.format(
                    np.mean(log['train_acc']),
                    np.mean(log['test_acc'])))
            print('--Classes: [training: {:s}, testing: {:s}]'.format(
                    ', '.join(['{:d}'.format(n) for n in log['train_cls']]),
                    ', '.join(['{:d}'.format(n) for n in log['test_cls']])))
            print('--Samples: [training: {:s}, testing: {:s}]'.format(
                    ', '.join(['{:d}'.format(n) for n in log['train_smpl']]),
                    ', '.join(['{:d}'.format(n) for n in log['test_smpl']])))
        else:
            dec.hit = None
            dec.intervals = None
            dec.stationary = None
            dec.running = None
            dec.high_state = None
            dec.low_state = None
            dec.sample_idxs = None
            dec.mdl_params = None

def natural_scene_image_decoding(sess_id, resume=False):

    def save_data_to_file(varData, varName):
        sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)

        fName = getattr(settings.projectData.files.sessions, varName)
        fPath = sessDir / fName
        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(varData, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

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

    varName = 'CascadeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        CascadeData = pickle.load(handle)

    varName = 'BehData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        BehData = pickle.load(handle)

    varName = 'SpikeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SpikeData = pickle.load(handle)

    varName = 'NeuEsmbl'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        NeuEsmbl = pickle.load(handle)

    if (not hasattr(CascadeData, 'diff_thr_from_GM')):
        print('the subject has no cascade data, skip ...')
        ImgDecoderData1 = DataContainer()
        save_data_to_file(ImgDecoderData1, 'ImgDecoderData1')

        ImgDecoderData2 = DataContainer()
        save_data_to_file(ImgDecoderData2, 'ImgDecoderData2')

        ImgDecoderData3 = DataContainer()
        save_data_to_file(ImgDecoderData3, 'ImgDecoderData3')
        return

    # state index
    df = imgNeuData.session_all.stimTable.loc[imgNeuData.session_all.imgStimID][['start_time']]
    df['stop_time'] = df['start_time'] + 0.2
    tbins = df.values.reshape(-1)

    intp_tres = 0.02
    intp_tpad = 1
    intp_time = np.arange(tbins[0]-intp_tpad, tbins[-1]+intp_tpad, intp_tres)

    spike_abs_time = SpikeData.spike_cnt.time_relative_to_stimulus_onset.values + SpikeData.start_time
    state_index_intp = np.interp(intp_time, spike_abs_time, CascadeData.p2n_spkdiff.values)
    rtns = stats.binned_statistic(intp_time, state_index_intp, statistic=np.nanmean, bins=tbins)

    bidx = np.arange(0, tbins.shape[0], 2)
    state_index = rtns.statistic[bidx]
    imgNeuData.session_all.state_index = state_index

    # stationary mask
    rtns = stats.binned_statistic(BehData.stationary_mask.time, BehData.stationary_mask.values, 
                                statistic='mean', bins=tbins)
    bidx = np.arange(0, tbins.shape[0], 2)
    stationary_mask = rtns.statistic[bidx]
    stationary_mask = np.floor(stationary_mask).astype(bool)
    imgNeuData.session_all.stationary_mask = stationary_mask

    # running mask
    rtns = stats.binned_statistic(BehData.running_mask.time, BehData.running_mask.values, 
                                statistic='mean', bins=tbins)
    bidx = np.arange(0, tbins.shape[0], 2)
    running_mask = rtns.statistic[bidx]
    running_mask = np.floor(running_mask).astype(bool)
    imgNeuData.session_all.running_mask = running_mask

    clsId2sampleIdx = {}
    for idx, imid in enumerate(imgNeuData.session_all.image_id):
        if imid in clsId2sampleIdx.keys():
            clsId2sampleIdx[imid].append(idx)
        else:
            clsId2sampleIdx[imid] = [idx]

    imgNeuData.session_all.clsId2sampleIdx = clsId2sampleIdx

    cfg = DataContainer()

    cfg.num_repeats = 75
    cfg.num_samples = imgNeuData.session_all.rsp_spikecnt_200ms.shape[0]
    cfg.num_units = imgNeuData.session_all.rsp_spikecnt_200ms.shape[1]
    cfg.stim_clsIds = sorted(clsId2sampleIdx.keys())[1:]
    cfg.min_repeats_per_class = 5

    RspData = DataContainer()

    RspData.spike_cnt = []
    RspData.spike_cnt_bsl = []
    RspData.invalid = []
    RspData.stationary = []
    RspData.running = []
    RspData.intervals = []
    RspData.targets = []
    RspData.state_index = []
    RspData.sample_idxs = []

    for cid in cfg.stim_clsIds:

        smpl_idxs = imgNeuData.session_all.clsId2sampleIdx[cid]
        nfill = cfg.num_repeats - len(smpl_idxs)

        spikecnt = imgNeuData.session_all.rsp_spikecnt_200ms[smpl_idxs, :].values / 2
        spikecnt = np.concatenate([spikecnt, np.full([nfill, cfg.num_units], np.nan)], axis=0)

        spikecnt_bsl = imgNeuData.session_all.bsl_spikecnt_500ms[smpl_idxs, :].values / 5
        spikecnt_bsl = np.concatenate([spikecnt_bsl, np.full([nfill, cfg.num_units], np.nan)], axis=0)

        invalid = imgNeuData.session_all.invalid[smpl_idxs]
        invalid = np.concatenate([invalid, np.full([nfill], True)])

        stationary = imgNeuData.session_all.stationary_mask[smpl_idxs]
        stationary = np.concatenate([stationary, np.full([nfill], False)])

        running = imgNeuData.session_all.running_mask[smpl_idxs]
        running = np.concatenate([running, np.full([nfill], False)])

        tgt = np.full([cfg.num_repeats], cid)

        start_time = imgNeuData.session_all.stimTable.iloc[smpl_idxs].start_time.values
        stop_time = imgNeuData.session_all.stimTable.iloc[smpl_idxs].stop_time.values
        intervals = np.stack([start_time, stop_time], axis=1)
        intervals = np.concatenate([intervals, np.full([nfill, 2], np.nan)])

        state = imgNeuData.session_all.state_index[smpl_idxs]
        state = np.concatenate([state, np.full([nfill], np.nan)])
        
        smpl_idxs = np.concatenate([smpl_idxs, np.full([nfill], np.nan)])

        RspData.spike_cnt.append(spikecnt)
        RspData.spike_cnt_bsl.append(spikecnt_bsl)
        RspData.invalid.append(invalid)
        RspData.stationary.append(stationary)
        RspData.running.append(running)
        RspData.intervals.append(intervals)
        RspData.targets.append(tgt)
        RspData.state_index.append(state)
        RspData.sample_idxs.append(smpl_idxs)

    spikecnt = np.stack(RspData.spike_cnt, axis=0)
    spikecnt = xr.DataArray(spikecnt, dims=['image_id', 'repeats', 'unit_id'],
                                    coords={'image_id': cfg.stim_clsIds,
                                            'repeats': np.arange(cfg.num_repeats),
                                            'unit_id': imgNeuData.session_all.rsp_spikecnt_200ms.unit_id.values})

    spikecnt_bsl = np.stack(RspData.spike_cnt_bsl, axis=0)
    spikecnt_bsl = xr.DataArray(spikecnt_bsl, coords=spikecnt.coords)

    RspData.spike_cnt = spikecnt.transpose('repeats', 'image_id', 'unit_id')
    RspData.spike_cnt_bsl = spikecnt_bsl.transpose('repeats', 'image_id', 'unit_id')
    RspData.invalid = np.stack(RspData.invalid, axis=0)
    RspData.stationary = np.stack(RspData.stationary, axis=0)
    RspData.running = np.stack(RspData.running, axis=0)
    RspData.intervals = np.stack(RspData.intervals, axis=0)
    RspData.targets = np.stack(RspData.targets, axis=0)
    RspData.state_index = np.stack(RspData.state_index, axis=0)
    RspData.sample_idxs = np.stack(RspData.sample_idxs, axis=0)

    spikecnt_mean = np.nanmean(SpikeData.spike_cnt, axis=0)
    spikecnt_std = np.nanstd(SpikeData.spike_cnt, axis=0)

    spikecnt_n  = RspData.spike_cnt.values.reshape(-1, cfg.num_units) / spikecnt_mean
    spikecnt_n = xr.DataArray(spikecnt_n.reshape(RspData.spike_cnt.shape), coords=RspData.spike_cnt.coords)

    RspData.spike_cnt_n = spikecnt_n

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        spikecnt_pos = spikecnt_n.sel(unit_id=SessData.units.index[CascadeData.posdly_umsk])
        spikecnt_neg = spikecnt_n.sel(unit_id=SessData.units.index[CascadeData.negdly_umsk])
        RspData.spkn_pos_mean = np.nanmean(spikecnt_pos, axis=2).T
        RspData.spkn_neg_mean = np.nanmean(spikecnt_neg, axis=2).T
        RspData.spkn_ratio = RspData.spkn_pos_mean / RspData.spkn_neg_mean

    RspData.response = RspData.spike_cnt - RspData.spike_cnt_bsl

    ImgDecoderData1 = DataContainer()
    ImgDecoderData2 = DataContainer()
    ImgDecoderData3 = DataContainer()

    opts_all = [
        # {'DecoderData': ImgDecoderData1,
        #  'DataName': 'ImgDecoderData1',
        #  'thr': CascadeData.diff_thr_from_GM, 
        #  'high_state_msk': RspData.state_index > CascadeData.diff_thr_from_GM,
        #  'low_state_msk': RspData.state_index < CascadeData.diff_thr_from_GM,
        #  'type': 'Neural Network',
        #  'classifier': 'NN'},

        {'DecoderData': ImgDecoderData2, 
         'DataName': 'ImgDecoderData2',
         'thr': CascadeData.diff_thr_from_GM, 
         'high_state_msk': RspData.state_index > CascadeData.diff_thr_from_GM,
         'low_state_msk': RspData.state_index < CascadeData.diff_thr_from_GM,
         'type': 'Logistic Regression',
         'classifier': 'LR'},

        {'DecoderData': ImgDecoderData3,
         'DataName': 'ImgDecoderData3', 
         'thr': CascadeData.diff_thr_from_GM, 
         'high_state_msk': RspData.state_index > CascadeData.diff_thr_from_GM,
         'low_state_msk': RspData.state_index < CascadeData.diff_thr_from_GM,
         'type': 'Support Vector Machine',
         'classifier': 'SVM'},
    ]

    for opt_dict in opts_all:

        ImgDecoderData = opt_dict['DecoderData']
        ratio_thr = opt_dict['thr']
        stype = opt_dict['type']

        RspData.high_state_msk = opt_dict['high_state_msk']
        RspData.low_state_msk = opt_dict['low_state_msk']

        """ K-Fold decoding performance evaluation for all samples """
        cfg.k = 5
        cfg.min_sample_per_fold = 5
        cfg.min_classes_per_fold = 50
        cfg.classifier = opt_dict['classifier']
        cfg.data_name = opt_dict['DataName']

        ImgDecoderData.cfg = cfg

        """ stationary high state """
        s = 'Stationary High State [{:s}, thr = {:.3f}, {:s}]'.format(stype, ratio_thr, cfg.classifier)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        ImgDecoderData.stationary_high = DataContainer()

        cond_msks = (~RspData.invalid) & RspData.high_state_msk & RspData.stationary
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)
        ImgDecoderData.stationary_high.kfTrain_msks = kfTrain_msks
        ImgDecoderData.stationary_high.kfTest_msks = kfTest_msks

        train_test_kFold(ImgDecoderData.stationary_high, NeuEsmbl, RspData, cfg)

        """ stationary low state """
        s = 'Stationary Low State [{:s}, thr = {:.3f}, {:s}]'.format(stype, ratio_thr, cfg.classifier)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        ImgDecoderData.stationary_low = DataContainer()

        cond_msks = (~RspData.invalid) & RspData.low_state_msk & RspData.stationary
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)
        ImgDecoderData.stationary_low.kfTrain_msks = kfTrain_msks
        ImgDecoderData.stationary_low.kfTest_msks = kfTest_msks

        train_test_kFold(ImgDecoderData.stationary_low, NeuEsmbl, RspData, cfg)

        """ running state """
        s = 'Running State [{:s}, thr = {:.3f}, {:s}]'.format(stype, ratio_thr, cfg.classifier)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        ImgDecoderData.running_all = DataContainer()
        cond_msks = (~RspData.invalid) & RspData.running
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)

        ImgDecoderData.running_all.kfTrain_msks = kfTrain_msks
        ImgDecoderData.running_all.kfTest_msks = kfTest_msks

        train_test_kFold(ImgDecoderData.running_all, NeuEsmbl, RspData, cfg)

        """ all data """
        s = 'All State [{:s}, thr = {:.3f}, {:s}]'.format(stype, ratio_thr, cfg.classifier)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        ImgDecoderData.all_state = DataContainer()
        cond_msks = ~RspData.invalid
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)

        ImgDecoderData.all_state.kfTrain_msks = kfTrain_msks
        ImgDecoderData.all_state.kfTest_msks = kfTest_msks

        train_test_kFold(ImgDecoderData.all_state, NeuEsmbl, RspData, cfg)

        """ save data """
        save_data_to_file(ImgDecoderData, cfg.data_name)


def natural_scene_image_decoding_excluding_regions(sess_id, resume=False):

    def save_data_to_file(ImgDecoderExRgData1, ImgDecoderExRgData2, ImgDecoderExRgData3):
        var2save = ['ImgDecoderExRgData1', 'ImgDecoderExRgData2', 'ImgDecoderExRgData3']
        sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
        for varName in var2save:
            fName = getattr(settings.projectData.files.sessions, varName)
            fPath = sessDir / fName

            with open(fPath, 'wb') as handle:
                print('save {:s} to {:s} ...'.format(varName, str(fPath)))
                pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('... done')

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

    varName = 'CascadeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        CascadeData = pickle.load(handle)

    varName = 'BehData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        BehData = pickle.load(handle)

    varName = 'SpikeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SpikeData = pickle.load(handle)

    varName = 'NeuEsmbl'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        NeuEsmbl = pickle.load(handle)

    if (not hasattr(CascadeData, 'diff_thr_from_GM')):
        print('the subject has no cascade data, skip ...')
        ImgDecoderExRgData1 = DataContainer()
        ImgDecoderExRgData2 = DataContainer()
        ImgDecoderExRgData3 = DataContainer()
        save_data_to_file(ImgDecoderExRgData1, ImgDecoderExRgData2, ImgDecoderExRgData3)
        return

    # state index
    df = imgNeuData.session_all.stimTable.loc[imgNeuData.session_all.imgStimID][['start_time']]
    df['stop_time'] = df['start_time'] + 0.2
    tbins = df.values.reshape(-1)

    intp_tres = 0.02
    intp_tpad = 1
    intp_time = np.arange(tbins[0]-intp_tpad, tbins[-1]+intp_tpad, intp_tres)

    spike_abs_time = SpikeData.spike_cnt.time_relative_to_stimulus_onset.values + SpikeData.start_time
    state_index_intp = np.interp(intp_time, spike_abs_time, CascadeData.p2n_spkdiff.values)
    rtns = stats.binned_statistic(intp_time, state_index_intp, statistic=np.nanmean, bins=tbins)

    bidx = np.arange(0, tbins.shape[0], 2)
    state_index = rtns.statistic[bidx]
    imgNeuData.session_all.state_index = state_index

    # stationary mask
    rtns = stats.binned_statistic(BehData.stationary_mask.time, BehData.stationary_mask.values, 
                                statistic='mean', bins=tbins)
    bidx = np.arange(0, tbins.shape[0], 2)
    stationary_mask = rtns.statistic[bidx]
    stationary_mask = np.floor(stationary_mask).astype(bool)
    imgNeuData.session_all.stationary_mask = stationary_mask

    # running mask
    rtns = stats.binned_statistic(BehData.running_mask.time, BehData.running_mask.values, 
                                statistic='mean', bins=tbins)
    bidx = np.arange(0, tbins.shape[0], 2)
    running_mask = rtns.statistic[bidx]
    running_mask = np.floor(running_mask).astype(bool)
    imgNeuData.session_all.running_mask = running_mask

    clsId2sampleIdx = {}
    for idx, imid in enumerate(imgNeuData.session_all.image_id):
        if imid in clsId2sampleIdx.keys():
            clsId2sampleIdx[imid].append(idx)
        else:
            clsId2sampleIdx[imid] = [idx]

    imgNeuData.session_all.clsId2sampleIdx = clsId2sampleIdx

    cfg = DataContainer()

    cfg.num_repeats = 75
    cfg.num_samples = imgNeuData.session_all.rsp_spikecnt_200ms.shape[0]
    cfg.num_units = imgNeuData.session_all.rsp_spikecnt_200ms.shape[1]
    cfg.stim_clsIds = sorted(clsId2sampleIdx.keys())[1:]
    cfg.min_repeats_per_class = 5

    RspData = DataContainer()

    RspData.spike_cnt = []
    RspData.spike_cnt_bsl = []
    RspData.invalid = []
    RspData.stationary = []
    RspData.running = []
    RspData.intervals = []
    RspData.targets = []
    RspData.state_index = []
    RspData.sample_idxs = []

    for cid in cfg.stim_clsIds:

        smpl_idxs = imgNeuData.session_all.clsId2sampleIdx[cid]
        nfill = cfg.num_repeats - len(smpl_idxs)

        spikecnt = imgNeuData.session_all.rsp_spikecnt_200ms[smpl_idxs, :].values / 2
        spikecnt = np.concatenate([spikecnt, np.full([nfill, cfg.num_units], np.nan)], axis=0)

        spikecnt_bsl = imgNeuData.session_all.bsl_spikecnt_500ms[smpl_idxs, :].values / 5
        spikecnt_bsl = np.concatenate([spikecnt_bsl, np.full([nfill, cfg.num_units], np.nan)], axis=0)

        invalid = imgNeuData.session_all.invalid[smpl_idxs]
        invalid = np.concatenate([invalid, np.full([nfill], True)])

        stationary = imgNeuData.session_all.stationary_mask[smpl_idxs]
        stationary = np.concatenate([stationary, np.full([nfill], False)])

        running = imgNeuData.session_all.running_mask[smpl_idxs]
        running = np.concatenate([running, np.full([nfill], False)])

        tgt = np.full([cfg.num_repeats], cid)

        start_time = imgNeuData.session_all.stimTable.iloc[smpl_idxs].start_time.values
        stop_time = imgNeuData.session_all.stimTable.iloc[smpl_idxs].stop_time.values
        intervals = np.stack([start_time, stop_time], axis=1)
        intervals = np.concatenate([intervals, np.full([nfill, 2], np.nan)])

        state = imgNeuData.session_all.state_index[smpl_idxs]
        state = np.concatenate([state, np.full([nfill], np.nan)])
        
        smpl_idxs = np.concatenate([smpl_idxs, np.full([nfill], np.nan)])

        RspData.spike_cnt.append(spikecnt)
        RspData.spike_cnt_bsl.append(spikecnt_bsl)
        RspData.invalid.append(invalid)
        RspData.stationary.append(stationary)
        RspData.running.append(running)
        RspData.intervals.append(intervals)
        RspData.targets.append(tgt)
        RspData.state_index.append(state)
        RspData.sample_idxs.append(smpl_idxs)

    spikecnt = np.stack(RspData.spike_cnt, axis=0)
    spikecnt = xr.DataArray(spikecnt, dims=['image_id', 'repeats', 'unit_id'],
                                    coords={'image_id': cfg.stim_clsIds,
                                            'repeats': np.arange(cfg.num_repeats),
                                            'unit_id': imgNeuData.session_all.rsp_spikecnt_200ms.unit_id.values})

    spikecnt_bsl = np.stack(RspData.spike_cnt_bsl, axis=0)
    spikecnt_bsl = xr.DataArray(spikecnt_bsl, coords=spikecnt.coords)

    RspData.spike_cnt = spikecnt.transpose('repeats', 'image_id', 'unit_id')
    RspData.spike_cnt_bsl = spikecnt_bsl.transpose('repeats', 'image_id', 'unit_id')
    RspData.invalid = np.stack(RspData.invalid, axis=0)
    RspData.stationary = np.stack(RspData.stationary, axis=0)
    RspData.running = np.stack(RspData.running, axis=0)
    RspData.intervals = np.stack(RspData.intervals, axis=0)
    RspData.targets = np.stack(RspData.targets, axis=0)
    RspData.state_index = np.stack(RspData.state_index, axis=0)
    RspData.sample_idxs = np.stack(RspData.sample_idxs, axis=0)

    spikecnt_mean = np.nanmean(SpikeData.spike_cnt, axis=0)
    spikecnt_std = np.nanstd(SpikeData.spike_cnt, axis=0)

    spikecnt_n  = RspData.spike_cnt.values.reshape(-1, cfg.num_units) / spikecnt_mean
    spikecnt_n = xr.DataArray(spikecnt_n.reshape(RspData.spike_cnt.shape), coords=RspData.spike_cnt.coords)

    RspData.spike_cnt_n = spikecnt_n

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        spikecnt_pos = spikecnt_n.sel(unit_id=SessData.units.index[CascadeData.posdly_umsk])
        spikecnt_neg = spikecnt_n.sel(unit_id=SessData.units.index[CascadeData.negdly_umsk])
        RspData.spkn_pos_mean = np.nanmean(spikecnt_pos, axis=2).T
        RspData.spkn_neg_mean = np.nanmean(spikecnt_neg, axis=2).T
        RspData.spkn_ratio = RspData.spkn_pos_mean / RspData.spkn_neg_mean

    RspData.response = RspData.spike_cnt - RspData.spike_cnt_bsl

    ImgDecoderExRgData1 = DataContainer()
    ImgDecoderExRgData2 = DataContainer()
    ImgDecoderExRgData3 = DataContainer()

    opts_all = [
        # {'DecoderData': ImgDecoderExRgData1, 
        #  'thr': CascadeData.diff_thr_from_GM, 
        #  'high_state_msk': RspData.state_index > CascadeData.diff_thr_from_GM,
        #  'low_state_msk': RspData.state_index < CascadeData.diff_thr_from_GM,
        #  'type': 'Neural Network',
        #  'classifier': 'NN'},

        # {'DecoderData': ImgDecoderExRgData2, 
        #  'thr': CascadeData.diff_thr_from_GM, 
        #  'high_state_msk': RspData.state_index > CascadeData.diff_thr_from_GM,
        #  'low_state_msk': RspData.state_index < CascadeData.diff_thr_from_GM,
        #  'type': 'Logistic Regression',
        #  'classifier': 'LR'},

        {'DecoderData': ImgDecoderExRgData3, 
         'thr': CascadeData.diff_thr_from_GM, 
         'high_state_msk': RspData.state_index > CascadeData.diff_thr_from_GM,
         'low_state_msk': RspData.state_index < CascadeData.diff_thr_from_GM,
         'type': 'Support Vector Machine',
         'classifier': 'SVM'},
    ]

    for opt_dict in opts_all:

        ImgDecoderData = opt_dict['DecoderData']
        ratio_thr = opt_dict['thr']
        stype = opt_dict['type']

        RspData.high_state_msk = opt_dict['high_state_msk']
        RspData.low_state_msk = opt_dict['low_state_msk']

        """ K-Fold decoding performance evaluation for all samples """
        cfg.k = 5
        cfg.min_sample_per_fold = 5
        cfg.min_classes_per_fold = 50
        cfg.classifier = opt_dict['classifier']

        ImgDecoderData.cfg = cfg

        """ all data """
        s = 'All State [{:s}, thr = {:.3f}]'.format(stype, ratio_thr)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        ImgDecoderData.all_state = DataContainer()
        cond_msks = ~RspData.invalid
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)

        ImgDecoderData.all_state.kfTrain_msks = kfTrain_msks
        ImgDecoderData.all_state.kfTest_msks = kfTest_msks

        train_test_kFold_exclude_regions(ImgDecoderData.all_state, NeuEsmbl, RspData, cfg)

    """ save data """
    save_data_to_file(ImgDecoderExRgData1, ImgDecoderExRgData2, ImgDecoderExRgData3)

def image_decoding_shuffled_control(sess_id, resume=False):

    def save_data_to_file(ImgDecoderCtrlData):
        var2save = ['ImgDecoderCtrlData']
        sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
        for varName in var2save:
            fName = getattr(settings.projectData.files.sessions, varName)
            fPath = sessDir / fName

            with open(fPath, 'wb') as handle:
                print('save {:s} to {:s} ...'.format(varName, str(fPath)))
                pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('... done')

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

    varName = 'CascadeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        CascadeData = pickle.load(handle)

    varName = 'BehData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        BehData = pickle.load(handle)

    varName = 'SpikeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SpikeData = pickle.load(handle)

    varName = 'NeuEsmbl'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        NeuEsmbl = pickle.load(handle)

    if (not hasattr(CascadeData, 'diff_thr_from_GM')):
        print('the subject has no cascade data, skip ...')
        ImgDecoderCtrlData = DataContainer()
        save_data_to_file(ImgDecoderCtrlData)
        return

    # state index
    df = imgNeuData.session_all.stimTable.loc[imgNeuData.session_all.imgStimID][['start_time']]
    df['stop_time'] = df['start_time'] + 0.2
    tbins = df.values.reshape(-1)

    intp_tres = 0.02
    intp_tpad = 1
    intp_time = np.arange(tbins[0]-intp_tpad, tbins[-1]+intp_tpad, intp_tres)

    spike_abs_time = SpikeData.spike_cnt.time_relative_to_stimulus_onset.values + SpikeData.start_time
    state_index_intp = np.interp(intp_time, spike_abs_time, CascadeData.p2n_spkdiff.values)
    rtns = stats.binned_statistic(intp_time, state_index_intp, statistic=np.nanmean, bins=tbins)

    bidx = np.arange(0, tbins.shape[0], 2)
    state_index = rtns.statistic[bidx]
    imgNeuData.session_all.state_index = state_index

    # stationary mask
    rtns = stats.binned_statistic(BehData.stationary_mask.time, BehData.stationary_mask.values, 
                                statistic='mean', bins=tbins)
    bidx = np.arange(0, tbins.shape[0], 2)
    stationary_mask = rtns.statistic[bidx]
    stationary_mask = np.floor(stationary_mask).astype(bool)
    imgNeuData.session_all.stationary_mask = stationary_mask

    # running mask
    rtns = stats.binned_statistic(BehData.running_mask.time, BehData.running_mask.values, 
                                statistic='mean', bins=tbins)
    bidx = np.arange(0, tbins.shape[0], 2)
    running_mask = rtns.statistic[bidx]
    running_mask = np.floor(running_mask).astype(bool)
    imgNeuData.session_all.running_mask = running_mask

    clsId2sampleIdx = {}
    for idx, imid in enumerate(imgNeuData.session_all.image_id):
        if imid in clsId2sampleIdx.keys():
            clsId2sampleIdx[imid].append(idx)
        else:
            clsId2sampleIdx[imid] = [idx]

    imgNeuData.session_all.clsId2sampleIdx = clsId2sampleIdx

    cfg = DataContainer()

    cfg.num_repeats = 75
    cfg.num_samples = imgNeuData.session_all.rsp_spikecnt_200ms.shape[0]
    cfg.num_units = imgNeuData.session_all.rsp_spikecnt_200ms.shape[1]
    cfg.stim_clsIds = sorted(clsId2sampleIdx.keys())[1:]
    cfg.min_repeats_per_class = 5

    RspData = DataContainer()

    RspData.spike_cnt = []
    RspData.spike_cnt_bsl = []
    RspData.invalid = []
    RspData.stationary = []
    RspData.running = []
    RspData.intervals = []
    RspData.targets = []
    RspData.state_index = []
    RspData.sample_idxs = []

    for cid in cfg.stim_clsIds:

        smpl_idxs = imgNeuData.session_all.clsId2sampleIdx[cid]
        nfill = cfg.num_repeats - len(smpl_idxs)

        spikecnt = imgNeuData.session_all.rsp_spikecnt_200ms[smpl_idxs, :].values / 2
        spikecnt = np.concatenate([spikecnt, np.full([nfill, cfg.num_units], np.nan)], axis=0)

        spikecnt_bsl = imgNeuData.session_all.bsl_spikecnt_500ms[smpl_idxs, :].values / 5
        spikecnt_bsl = np.concatenate([spikecnt_bsl, np.full([nfill, cfg.num_units], np.nan)], axis=0)

        invalid = imgNeuData.session_all.invalid[smpl_idxs]
        invalid = np.concatenate([invalid, np.full([nfill], True)])

        stationary = imgNeuData.session_all.stationary_mask[smpl_idxs]
        stationary = np.concatenate([stationary, np.full([nfill], False)])

        running = imgNeuData.session_all.running_mask[smpl_idxs]
        running = np.concatenate([running, np.full([nfill], False)])

        tgt = np.full([cfg.num_repeats], cid)

        start_time = imgNeuData.session_all.stimTable.iloc[smpl_idxs].start_time.values
        stop_time = imgNeuData.session_all.stimTable.iloc[smpl_idxs].stop_time.values
        intervals = np.stack([start_time, stop_time], axis=1)
        intervals = np.concatenate([intervals, np.full([nfill, 2], np.nan)])

        state = imgNeuData.session_all.state_index[smpl_idxs]
        state = np.concatenate([state, np.full([nfill], np.nan)])
        
        smpl_idxs = np.concatenate([smpl_idxs, np.full([nfill], np.nan)])

        RspData.spike_cnt.append(spikecnt)
        RspData.spike_cnt_bsl.append(spikecnt_bsl)
        RspData.invalid.append(invalid)
        RspData.stationary.append(stationary)
        RspData.running.append(running)
        RspData.intervals.append(intervals)
        RspData.targets.append(tgt)
        RspData.state_index.append(state)
        RspData.sample_idxs.append(smpl_idxs)

    spikecnt = np.stack(RspData.spike_cnt, axis=0)
    spikecnt = xr.DataArray(spikecnt, dims=['image_id', 'repeats', 'unit_id'],
                                    coords={'image_id': cfg.stim_clsIds,
                                            'repeats': np.arange(cfg.num_repeats),
                                            'unit_id': imgNeuData.session_all.rsp_spikecnt_200ms.unit_id.values})

    spikecnt_bsl = np.stack(RspData.spike_cnt_bsl, axis=0)
    spikecnt_bsl = xr.DataArray(spikecnt_bsl, coords=spikecnt.coords)

    RspData.spike_cnt = spikecnt.transpose('repeats', 'image_id', 'unit_id')
    RspData.spike_cnt_bsl = spikecnt_bsl.transpose('repeats', 'image_id', 'unit_id')
    RspData.invalid = np.stack(RspData.invalid, axis=0)
    RspData.stationary = np.stack(RspData.stationary, axis=0)
    RspData.running = np.stack(RspData.running, axis=0)
    RspData.intervals = np.stack(RspData.intervals, axis=0)
    RspData.targets = np.stack(RspData.targets, axis=0)
    RspData.state_index = np.stack(RspData.state_index, axis=0)
    RspData.sample_idxs = np.stack(RspData.sample_idxs, axis=0)

    spikecnt_mean = np.nanmean(SpikeData.spike_cnt, axis=0)
    spikecnt_std = np.nanstd(SpikeData.spike_cnt, axis=0)

    spikecnt_n  = RspData.spike_cnt.values.reshape(-1, cfg.num_units) / spikecnt_mean
    spikecnt_n = xr.DataArray(spikecnt_n.reshape(RspData.spike_cnt.shape), coords=RspData.spike_cnt.coords)

    RspData.spike_cnt_n = spikecnt_n

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        spikecnt_pos = spikecnt_n.sel(unit_id=SessData.units.index[CascadeData.posdly_umsk])
        spikecnt_neg = spikecnt_n.sel(unit_id=SessData.units.index[CascadeData.negdly_umsk])
        RspData.spkn_pos_mean = np.nanmean(spikecnt_pos, axis=2).T
        RspData.spkn_neg_mean = np.nanmean(spikecnt_neg, axis=2).T
        RspData.spkn_ratio = RspData.spkn_pos_mean / RspData.spkn_neg_mean

    RspData.response = RspData.spike_cnt - RspData.spike_cnt_bsl

    """ randomly shuffle the neural activities """
    rng = np.random.default_rng(12345)
    N = RspData.spike_cnt.values.reshape(-1, cfg.num_units).shape[0]
    sample_idx_pm = rng.permutation(N)

    d1, d2, d3 = RspData.spike_cnt.shape
    spike_cnt_pm = RspData.spike_cnt.values.reshape(-1, d3)[sample_idx_pm, :].reshape(d1, d2, d3)
    RspData.spike_cnt = xr.DataArray(spike_cnt_pm, coords=RspData.spike_cnt.coords)

    spike_cnt_n_pm = RspData.spike_cnt_n.values.reshape(-1, d3)[sample_idx_pm, :].reshape(d1, d2, d3)
    RspData.spike_cnt_n = xr.DataArray(spike_cnt_n_pm, coords=RspData.spike_cnt_n.coords)

    spike_cnt_bsl_pm = RspData.spike_cnt_bsl.values.reshape(-1, d3)[sample_idx_pm, :].reshape(d1, d2, d3)
    RspData.spike_cnt_bsl = xr.DataArray(spike_cnt_bsl_pm, coords=RspData.spike_cnt_bsl.coords)

    response_pm = RspData.response.values.reshape(-1, d3)[sample_idx_pm, :].reshape(d1, d2, d3)
    RspData.response = xr.DataArray(response_pm, coords=RspData.response.coords)

    RspData.invalid = RspData.invalid.T.reshape(-1)[sample_idx_pm].reshape(d1, d2).T
    RspData.stationary = RspData.stationary.T.reshape(-1)[sample_idx_pm].reshape(d1, d2).T
    RspData.running = RspData.running.T.reshape(-1)[sample_idx_pm].reshape(d1, d2).T
    RspData.state_index = RspData.state_index.T.reshape(-1)[sample_idx_pm].reshape(d1, d2).T
    RspData.sample_idxs = RspData.sample_idxs.T.reshape(-1)[sample_idx_pm].reshape(d1, d2).T
    RspData.spkn_ratio = RspData.spkn_ratio.T.reshape(-1)[sample_idx_pm].reshape(d1, d2).T
    RspData.intervals = RspData.intervals.transpose([1,0,2]).reshape(-1, 2)[sample_idx_pm, :].reshape(d1, d2, 2).transpose([1,0,2])

    ImgDecoderCtrlData = DataContainer()

    opts_all = [
        {'DecoderData': ImgDecoderCtrlData, 
         'thr': CascadeData.diff_thr_from_GM, 
         'high_state_msk': RspData.state_index > CascadeData.diff_thr_from_GM,
         'low_state_msk': RspData.state_index < CascadeData.diff_thr_from_GM,
         'type': 'Support Vector Machine',
         'classifier': 'SVM'},
    ]

    for opt_dict in opts_all:

        ImgDecoderData = opt_dict['DecoderData']
        ratio_thr = opt_dict['thr']
        stype = opt_dict['type']

        RspData.high_state_msk = opt_dict['high_state_msk']
        RspData.low_state_msk = opt_dict['low_state_msk']

        """ K-Fold decoding performance evaluation for all samples """
        cfg.k = 5
        cfg.min_sample_per_fold = 5
        cfg.min_classes_per_fold = 50
        cfg.classifier = opt_dict['classifier']

        ImgDecoderData.cfg = cfg

        """ all data """
        s = 'All State [{:s}, thr = {:.3f}]'.format(stype, ratio_thr)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        ImgDecoderData.all_state = DataContainer()
        cond_msks = ~RspData.invalid
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)

        ImgDecoderData.all_state.kfTrain_msks = kfTrain_msks
        ImgDecoderData.all_state.kfTest_msks = kfTest_msks

        train_test_kFold(ImgDecoderData.all_state, NeuEsmbl, RspData, cfg)

    """ save data """
    save_data_to_file(ImgDecoderCtrlData)

def get_DG_classes(dgNeuData, type='ori'):
    if type == 'ori':
        """ only orientation """
        stim_name2id = {}
        name_tmpl = '({:s})'

        cid = 0
        for ori in dgNeuData.session_all.dg_properties.orientation.unique():
            key = name_tmpl.format(str(ori))
            stim_name2id[key] = cid

            if ori == 'null':
                stim_name2id[key] = -1
            else:
                cid += 1

        sample_clsIds = []
        for (ori, tfreq, ctrst) in dgNeuData.session_all.dg_properties.values:
            name = name_tmpl.format(str(ori))
            sample_clsIds.append(stim_name2id[name])

    elif type == 'tfreq':
        """ only temporal frequency """
        stim_name2id = {}
        name_tmpl = '({:s})'

        cid = 0
        for tfreq in dgNeuData.session_all.dg_properties.temporal_frequency.unique():
            key = name_tmpl.format(str(tfreq))
            stim_name2id[key] = cid
                
            if tfreq == 'null':
                stim_name2id[key] = -1
            else:
                cid += 1
                
        sample_clsIds = []
        for (ori, tfreq, ctrst) in dgNeuData.session_all.dg_properties.values:
            name = name_tmpl.format(str(tfreq))
            sample_clsIds.append(stim_name2id[name])

    else:
        """ both orientation and temporal frequency """
        stim_name2id = {}
        name_tmpl = '({:s}, {:s})'

        cid = 0
        for ori in dgNeuData.session_all.dg_properties.orientation.unique():
            msk = dgNeuData.session_all.dg_properties.orientation.isin([ori])
            for tfreq in dgNeuData.session_all.dg_properties.temporal_frequency[msk].unique():
                key = name_tmpl.format(str(ori), str(tfreq))
                stim_name2id[key] = cid
                
                if ori == 'null':
                    stim_name2id[key] = -1
                else:
                    cid += 1

        sample_clsIds = []
        for (ori, tfreq, ctrst) in dgNeuData.session_all.dg_properties.values:
            name = name_tmpl.format(str(ori), str(tfreq))
            sample_clsIds.append(stim_name2id[name])

    clsId2sampleIdx = {}
    for idx, cid in enumerate(sample_clsIds):
        if cid in clsId2sampleIdx.keys():
            clsId2sampleIdx[cid].append(idx)
        else:
            clsId2sampleIdx[cid] = [idx]
            
    return stim_name2id, clsId2sampleIdx, sample_clsIds

def drifting_grating_decoding(sess_id, resume=False):

    def save_data_to_file(DgDecoderData1, DgDecoderData2, DgDecoderData3):
        var2save = ['DgDecoderData1', 'DgDecoderData2', 'DgDecoderData3']
        sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
        for varName in var2save:
            fName = getattr(settings.projectData.files.sessions, varName)
            fPath = sessDir / fName

            with open(fPath, 'wb') as handle:
                print('save {:s} to {:s} ...'.format(varName, str(fPath)))
                pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('... done')

    def _get_decoder_data(cfg):
        stim_name2id, clsId2sampleIdx, sample_clsIds = get_DG_classes(dgNeuData, type=cfg.dg_type)
        dgNeuData.session_all.clsId2sampleIdx = clsId2sampleIdx
        cfg.stim_clsIds = sorted(clsId2sampleIdx.keys())[1:]

        RspData = DataContainer()

        RspData.spike_cnt = []
        RspData.spike_cnt_bsl = []
        RspData.invalid = []
        RspData.stationary = []
        RspData.running = []
        RspData.intervals = []
        RspData.targets = []
        RspData.state_index = []
        RspData.sample_idxs = []

        for cid in cfg.stim_clsIds:

            smpl_idxs = dgNeuData.session_all.clsId2sampleIdx[cid]
            nfill = cfg.num_repeats - len(smpl_idxs)

            spikecnt = dgNeuData.session_all.rsp_spikecnt_200ms[smpl_idxs, :].values / 2
            spikecnt = np.concatenate([spikecnt, np.full([nfill, cfg.num_units], np.nan)], axis=0)

            spikecnt_bsl = dgNeuData.session_all.bsl_spikecnt_800ms[smpl_idxs, :].values / 8
            spikecnt_bsl = np.concatenate([spikecnt_bsl, np.full([nfill, cfg.num_units], np.nan)], axis=0)

            invalid = dgNeuData.session_all.invalid[smpl_idxs]
            invalid = np.concatenate([invalid, np.full([nfill], True)])

            stationary = dgNeuData.session_all.stationary_mask[smpl_idxs]
            stationary = np.concatenate([stationary, np.full([nfill], False)])

            running = dgNeuData.session_all.running_mask[smpl_idxs]
            running = np.concatenate([running, np.full([nfill], False)])

            tgt = np.full([cfg.num_repeats], cid)

            start_time = dgNeuData.session_all.stimTable.iloc[smpl_idxs].start_time.values
            stop_time = dgNeuData.session_all.stimTable.iloc[smpl_idxs].stop_time.values
            intervals = np.stack([start_time, stop_time], axis=1)
            intervals = np.concatenate([intervals, np.full([nfill, 2], np.nan)])

            state = dgNeuData.session_all.state_index[smpl_idxs]
            state = np.concatenate([state, np.full([nfill], np.nan)])

            smpl_idxs = np.concatenate([smpl_idxs, np.full([nfill], np.nan)])

            RspData.spike_cnt.append(spikecnt)
            RspData.spike_cnt_bsl.append(spikecnt_bsl)
            RspData.invalid.append(invalid)
            RspData.stationary.append(stationary)
            RspData.running.append(running)
            RspData.intervals.append(intervals)
            RspData.targets.append(tgt)
            RspData.state_index.append(state)
            RspData.sample_idxs.append(smpl_idxs)

        spikecnt = np.stack(RspData.spike_cnt, axis=0)
        spikecnt = xr.DataArray(spikecnt, dims=['image_id', 'repeats', 'unit_id'],
                                        coords={'image_id': cfg.stim_clsIds,
                                                'repeats': np.arange(cfg.num_repeats),
                                                'unit_id': dgNeuData.session_all.rsp_spikecnt_200ms.unit_id.values})

        spikecnt_bsl = np.stack(RspData.spike_cnt_bsl, axis=0)
        spikecnt_bsl = xr.DataArray(spikecnt_bsl, coords=spikecnt.coords)

        RspData.spike_cnt = spikecnt.transpose('repeats', 'image_id', 'unit_id')
        RspData.spike_cnt_bsl = spikecnt_bsl.transpose('repeats', 'image_id', 'unit_id')
        RspData.invalid = np.stack(RspData.invalid, axis=0)
        RspData.stationary = np.stack(RspData.stationary, axis=0)
        RspData.running = np.stack(RspData.running, axis=0)
        RspData.intervals = np.stack(RspData.intervals, axis=0)
        RspData.targets = np.stack(RspData.targets, axis=0)
        RspData.state_index = np.stack(RspData.state_index, axis=0)
        RspData.sample_idxs = np.stack(RspData.sample_idxs, axis=0)

        spikecnt_mean = np.nanmean(SpikeData.spike_cnt, axis=0)
        spikecnt_std = np.nanstd(SpikeData.spike_cnt, axis=0)

        spikecnt_n  = RspData.spike_cnt.values.reshape(-1, cfg.num_units) / spikecnt_mean
        spikecnt_n = xr.DataArray(spikecnt_n.reshape(RspData.spike_cnt.shape), coords=RspData.spike_cnt.coords)

        RspData.spike_cnt_n = spikecnt_n

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            spikecnt_pos = spikecnt_n.sel(unit_id=SessData.units.index[CascadeData.posdly_umsk])
            spikecnt_neg = spikecnt_n.sel(unit_id=SessData.units.index[CascadeData.negdly_umsk])
            RspData.spkn_pos_mean = np.nanmean(spikecnt_pos, axis=2).T
            RspData.spkn_neg_mean = np.nanmean(spikecnt_neg, axis=2).T
            RspData.spkn_ratio = RspData.spkn_pos_mean / RspData.spkn_neg_mean

        RspData.response = RspData.spike_cnt - RspData.spike_cnt_bsl

        DgDecoderData = DataContainer()
        opt_dict = {
            'DecoderData': DgDecoderData, 
            'thr': CascadeData.diff_thr_from_GM, 
            'high_state_msk': RspData.state_index > CascadeData.diff_thr_from_GM,
            'low_state_msk': RspData.state_index < CascadeData.diff_thr_from_GM,
            'type': 'Support Vector Machine',
            'classifier': 'SVM'}
        
        DgDecoderData = opt_dict['DecoderData']
        ratio_thr = opt_dict['thr']
        stype = opt_dict['type']

        RspData.high_state_msk = opt_dict['high_state_msk']
        RspData.low_state_msk = opt_dict['low_state_msk']

        """ K-Fold decoding performance evaluation for all samples """
        cfg.classifier = opt_dict['classifier']

        DgDecoderData.cfg = cfg

        """ stationary high state """
        s = 'Stationary High State [{:s}, thr = {:.3f}, DG type: {:s}]'.format(stype, ratio_thr, cfg.dg_type)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        DgDecoderData.stationary_high = DataContainer()

        cond_msks = (~RspData.invalid) & RspData.high_state_msk & RspData.stationary
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)
        DgDecoderData.stationary_high.kfTrain_msks = kfTrain_msks
        DgDecoderData.stationary_high.kfTest_msks = kfTest_msks

        train_test_kFold(DgDecoderData.stationary_high, NeuEsmbl, RspData, cfg)

        """ stationary low state """
        s = 'Stationary Low State [{:s}, thr = {:.3f}, DG type: {:s}]'.format(stype, ratio_thr, cfg.dg_type)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        DgDecoderData.stationary_low = DataContainer()

        cond_msks = (~RspData.invalid) & RspData.low_state_msk & RspData.stationary
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)
        DgDecoderData.stationary_low.kfTrain_msks = kfTrain_msks
        DgDecoderData.stationary_low.kfTest_msks = kfTest_msks

        train_test_kFold(DgDecoderData.stationary_low, NeuEsmbl, RspData, cfg)

        """ running state """
        s = 'Running State [{:s}, thr = {:.3f}, DG type: {:s}]'.format(stype, ratio_thr, cfg.dg_type)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        DgDecoderData.running_all = DataContainer()
        cond_msks = (~RspData.invalid) & RspData.running
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)

        DgDecoderData.running_all.kfTrain_msks = kfTrain_msks
        DgDecoderData.running_all.kfTest_msks = kfTest_msks

        train_test_kFold(DgDecoderData.running_all, NeuEsmbl, RspData, cfg)

        """ all data """
        s = 'All State [{:s}, thr = {:.3f}, DG type: {:s}]'.format(stype, ratio_thr, cfg.dg_type)
        s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
        print(s)
        DgDecoderData.all_state = DataContainer()
        cond_msks = ~RspData.invalid
        kfTrain_msks, kfTest_msks = conditional_sample_mask_for_kfold_validation(cfg, RspData, cond_msks)

        DgDecoderData.all_state.kfTrain_msks = kfTrain_msks
        DgDecoderData.all_state.kfTest_msks = kfTest_msks

        train_test_kFold(DgDecoderData.all_state, NeuEsmbl, RspData, cfg)

        return DgDecoderData

    sessDir = settings.visOrg.dir.sessions / '{:d}'.format(sess_id)
    fDir = sessDir / settings.visOrg.ob_sess_rel_dir.drifting_gratings
    varName = 'dgNeuData'
    fName = getattr(settings.visOrg.files.sessions, varName)
    fpath = fDir / fName
    with open(fpath, 'rb') as handle:
        dgNeuData = pickle.load(handle)

    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    varName = 'CascadeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        CascadeData = pickle.load(handle)

    varName = 'BehData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        BehData = pickle.load(handle)

    varName = 'SpikeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SpikeData = pickle.load(handle)

    varName = 'NeuEsmbl'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        NeuEsmbl = pickle.load(handle)

    if (not hasattr(CascadeData, 'diff_thr_from_GM')):
        print('the subject has no cascade data, skip ...')
        DgDecoderData1 = DataContainer()
        DgDecoderData2 = DataContainer()
        DgDecoderData3 = DataContainer()
        save_data_to_file(DgDecoderData1, DgDecoderData2, DgDecoderData3)
        return

    # state index
    df = dgNeuData.session_all.stimTable.loc[dgNeuData.session_all.imgStimID][['start_time']]
    df['stop_time'] = df.start_time + 0.2
    tbins = df.values.reshape(-1)

    intp_tres = 0.02
    intp_tpad = 1
    intp_time = np.arange(tbins[0]-intp_tpad, tbins[-1]+intp_tpad, intp_tres)

    spike_abs_time = SpikeData.spike_cnt.time_relative_to_stimulus_onset.values + SpikeData.start_time
    state_index_intp = np.interp(intp_time, spike_abs_time, CascadeData.p2n_spkdiff.values)
    rtns = stats.binned_statistic(intp_time, state_index_intp, statistic=np.nanmean, bins=tbins)

    bidx = np.arange(0, tbins.shape[0], 2)
    state_index = rtns.statistic[bidx]
    dgNeuData.session_all.state_index = state_index

    # stationary mask
    rtns = stats.binned_statistic(BehData.stationary_mask.time, BehData.stationary_mask.values, 
                                statistic='mean', bins=tbins)
    bidx = np.arange(0, tbins.shape[0], 2)
    stationary_mask = rtns.statistic[bidx]
    stationary_mask = np.floor(stationary_mask).astype(bool)
    dgNeuData.session_all.stationary_mask = stationary_mask

    # running mask
    rtns = stats.binned_statistic(BehData.running_mask.time, BehData.running_mask.values, 
                                statistic='mean', bins=tbins)
    bidx = np.arange(0, tbins.shape[0], 2)
    running_mask = rtns.statistic[bidx]
    running_mask = np.floor(running_mask).astype(bool)
    dgNeuData.session_all.running_mask = running_mask

    """ DG stimulus type: orientation """
    cfg = DataContainer()
    cfg.num_repeats = 100 # 75 in total
    cfg.num_samples = dgNeuData.session_all.rsp_spikecnt_200ms.shape[0]
    cfg.num_units = dgNeuData.session_all.rsp_spikecnt_200ms.shape[1]
    cfg.min_repeats_per_class = 5
    cfg.k = 5
    cfg.min_sample_per_fold = 5
    cfg.min_classes_per_fold = 5 # 8 in total
    cfg.dg_type = 'ori'

    DgDecoderData1 = _get_decoder_data(cfg)

    """ DG stimulus type: temporal frequency """
    cfg = DataContainer()
    cfg.num_repeats = 150 # 120 in total
    cfg.num_samples = dgNeuData.session_all.rsp_spikecnt_200ms.shape[0]
    cfg.num_units = dgNeuData.session_all.rsp_spikecnt_200ms.shape[1]
    cfg.min_repeats_per_class = 5 
    cfg.k = 5
    cfg.min_sample_per_fold = 5
    cfg.min_classes_per_fold = 5 # 5 in total
    cfg.dg_type = 'tfreq'

    DgDecoderData2 = _get_decoder_data(cfg)

    """ DG stimulus type: orientation + temporal frequency"""
    cfg = DataContainer()
    cfg.num_repeats = 30 # 15 in total
    cfg.num_samples = dgNeuData.session_all.rsp_spikecnt_200ms.shape[0]
    cfg.num_units = dgNeuData.session_all.rsp_spikecnt_200ms.shape[1]
    cfg.min_repeats_per_class = 3
    cfg.k = 5
    cfg.min_sample_per_fold = 5
    cfg.min_classes_per_fold = 10 # 40 in total
    cfg.dg_type = 'both'

    DgDecoderData3 = _get_decoder_data(cfg)

    save_data_to_file(DgDecoderData1, DgDecoderData2, DgDecoderData3)

def proc_decoding_natural_scene_image(resume=False):
    sessions2proc = session_ob_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing natural scene image decoding for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))

        natural_scene_image_decoding(sess_id, resume=resume)

def proc_decoding_natural_scene_image_excluding_regions(resume=False):
    sessions2proc = session_ob_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing natural scene image decoding [importance check] for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))

        natural_scene_image_decoding_excluding_regions(sess_id, resume=resume)

def proc_decoding_natural_scene_image_shuffled_control(resume=False):
    sessions2proc = session_ob_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing natural scene image decoding [shuffled control] for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))

        image_decoding_shuffled_control(sess_id, resume=resume)

def proc_decoding_drifting_grating(resume=False):
    sessions2proc = session_ob_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing drifting-grating response for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))

        drifting_grating_decoding(sess_id, resume=resume)

def main():
    # proc_decoding_natural_scene_image(resume=False)

    # proc_decoding_natural_scene_image_excluding_regions(resume=False)

    proc_decoding_natural_scene_image_shuffled_control(resume=False)

    proc_decoding_drifting_grating(resume=False)

if __name__ == "__main__":
    main()