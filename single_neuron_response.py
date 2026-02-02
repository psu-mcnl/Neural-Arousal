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
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

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

def get_response_for_significant_stim_unit_pair(RspData, cfg, cond_msks):
    snr_all = []
    rsp_all = []
    rspn_all = []
    rspn2_all = []
    cid_all = []
    stim_pid_all = []
    uid_all = []
    msign_all = []
    tscore_all = []
    ratio_all = []
    diff_all = []
    n_samples = []
    state_all = []

    running_all = []
    stationary_all = []

    for ui in RspData.stim_sel_msk.reshape(-1, cfg.num_units).any(axis=0).nonzero()[0]:
        for ci in RspData.stim_sel_msk[:,:,ui].any(axis=0).nonzero()[0]:
            msk = cond_msks & RspData.stim_sel_msk

            ucmsk = np.full([len(cfg.stim_clsIds), cfg.num_units], False)
            ucmsk[ci,ui] = True
            msk[:,~ucmsk] = False
            if msk.sum() > cfg.min_sample_per_condition:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    spk_stim = np.nanmean(RspData.spike_cnt_ost.values[msk])
                    spk_bsl = np.nanmean(RspData.spike_cnt_bsl.values[msk])
                    snr = (spk_stim - spk_bsl) / (spk_stim + spk_bsl)

                    rsp = RspData.response.values[msk]
                    rsp_n = RspData.response_n.values[msk]
                    rsp_n2 = RspData.response_n2.values[msk]
                    cid = np.unique(RspData.stim_ids[msk]).squeeze()
                    uid = np.unique(RspData.unit_ids[msk]).squeeze()
                    pid = RspData.stim_pid[msk]
                    msign = np.unique(RspData.stim_mod_sign[msk]).squeeze()
                    tscore = np.unique(RspData.stim_tscores[msk]).squeeze()
                    ratio = RspData.spkn_ratio[msk]
                    diff = RspData.spkn_diff[msk]
                    state = RspData.state_index[msk]
                    ns = msk.sum()

                    running = RspData.running[msk]
                    stationary = RspData.stationary[msk]
            else:
                rsp = np.array([])
                rsp_n = np.array([])
                rsp_n2 = np.array([])
                cid = np.nan
                uid = np.nan
                pid = np.array([])
                msign = np.nan
                tscore = np.nan
                ratio = np.array([])
                diff = np.array([])
                state = np.array([])
                snr = np.nan
                ns = np.nan

                running = np.array([])
                stationary = np.array([])

            snr_all.append(snr)
            rsp_all.append(rsp)
            rspn_all.append(rsp_n)
            rspn2_all.append(rsp_n2)
            cid_all.append(cid)
            stim_pid_all.append(pid)
            uid_all.append(uid)
            msign_all.append(msign)
            tscore_all.append(tscore)
            ratio_all.append(ratio)
            diff_all.append(diff)
            state_all.append(state)
            n_samples.append(ns)

            running_all.append(running)
            stationary_all.append(stationary)
            
    outputs = DataContainer()
    outputs.rsps = rsp_all
    outputs.rsps_n = rspn_all
    outputs.rsps_n2 = rspn2_all
    outputs.p2n_ratios = ratio_all
    outputs.p2n_diff = diff_all
    outputs.state_indexes = state_all
    outputs.snrs = np.array(snr_all)
    outputs.class_ids = np.array(cid_all)
    outputs.stim_presentation_ids = stim_pid_all
    outputs.unit_ids = np.array(uid_all)
    outputs.mod_sign = np.array(msign_all)
    outputs.mod_scores = np.array(tscore_all)
    outputs.n_samples = np.array(n_samples)

    outputs.running = running_all
    outputs.stationary = stationary_all
    return outputs

def get_response_for_control_stim_unit_pair(RspData, cfg, cond_msks):
    rsp_ctl_all = []
    rspn_ctl_all = []
    rspn2_ctl_all = []
    uid_ctl_all = []
    stim_pid_ctl_all = []
    cid_all = []
    msign_all = []
    tscore_all = []
    n_samples_ctl = []

    # only compute for units with selective responses
    for ui in RspData.stim_ctl_msk.reshape(-1, cfg.num_units).any(axis=0).nonzero()[0]:
        for ci in RspData.stim_ctl_msk[:,:,ui].any(axis=0).nonzero()[0]:
            msk = cond_msks & RspData.stim_ctl_msk

            ucmsk = np.full([len(cfg.stim_clsIds), cfg.num_units], False)
            ucmsk[ci,ui] = True
            msk[:,~ucmsk] = False
            
            if msk.sum() > cfg.min_sample_per_condition:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    
                    rsp = RspData.response.values[msk]
                    rsp_n = RspData.response_n.values[msk]
                    rsp_n2 = RspData.response_n2.values[msk]
                    uid = np.unique(RspData.unit_ids[msk]).squeeze()
                    pid = RspData.stim_pid[msk]
                    cid = np.unique(RspData.stim_ids[msk]).squeeze()
                    msign = np.unique(RspData.stim_mod_sign[msk]).squeeze()
                    tscore = np.unique(RspData.stim_tscores[msk]).squeeze()
                    ns = msk.sum()
                    
            else:
                rsp = np.array([])
                rsp_n = np.array([])
                rsp_n2 = np.array([])
                uid = np.nan
                pid = np.array([])
                cid = np.nan
                msign = np.nan
                tscore = np.nan
                ns = np.nan
        
            rsp_ctl_all.append(rsp)
            rspn_ctl_all.append(rsp_n)
            rspn2_ctl_all.append(rsp_n2)
            uid_ctl_all.append(uid)
            stim_pid_ctl_all.append(pid)
            cid_all.append(cid)
            msign_all.append(msign)
            tscore_all.append(tscore)
            n_samples_ctl.append(ns)

    outputs = DataContainer()
    outputs.rsps_ctrl = rsp_ctl_all
    outputs.rsps_n_ctrl = rspn_ctl_all
    outputs.rsps_n2_ctrl = rspn2_ctl_all
    outputs.unit_ids = np.array(uid_ctl_all)
    outputs.stim_presentation_ids = stim_pid_ctl_all
    outputs.class_ids = np.array(cid_all)
    outputs.mod_sign = np.array(msign_all)
    outputs.mod_scores = np.array(tscore_all)
    outputs.n_samples = np.array(n_samples_ctl)
    return outputs

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

def compute_single_neuron_DG_response_index(sess_id, resume=False):

    def save_data_to_file(DgRspData1):
        var2save = ['DgRspData1']
        sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
        for varName in var2save:
            fName = getattr(settings.projectData.files.sessions, varName)
            fPath = sessDir / fName

            with open(fPath, 'wb') as handle:
                print('save {:s} to {:s} ...'.format(varName, str(fPath)))
                pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('... done')

    def _get_DG_response(cfg):
        RspData = DataContainer()

        RspData.spike_cnt_ost = []
        RspData.spike_cnt_stm = []
        RspData.spike_cnt_pre = []
        RspData.spike_cnt_bsl = []
        RspData.invalid = []
        RspData.stationary = []
        RspData.running = []
        RspData.intervals = []
        RspData.targets = []
        RspData.stim_pid = [] # stimulus presentation id
        RspData.state_index = []

        for cid in cfg.stim_clsIds:

            smpl_idxs = cfg.clsId2sampleIdx[cid]
            nfill = cfg.num_repeats - len(smpl_idxs)

            spikecnt_ost = dgNeuData.session_all.rsp_spikecnt_400ms[smpl_idxs, :].values / 2
            spikecnt_ost = np.concatenate([spikecnt_ost, np.full([nfill, cfg.num_units], np.nan)], axis=0)
            
            spikecnt_stm = dgNeuData.session_all.rsp_spikecnt_600ms[smpl_idxs, :].values / 3
            spikecnt_stm = np.concatenate([spikecnt_stm, np.full([nfill, cfg.num_units], np.nan)], axis=0)
            
            spikecnt_pre = (dgNeuData.session_all.bsl_spikecnt_200ms[smpl_idxs, :].values + \
                            dgNeuData.session_all.rsp_spikecnt_200ms[smpl_idxs, :].values) / 2
            spikecnt_pre = np.concatenate([spikecnt_pre, np.full([nfill, cfg.num_units], np.nan)], axis=0)

            spikecnt_bsl = dgNeuData.session_all.bsl_spikecnt_800ms[smpl_idxs, :].values / 4
            spikecnt_bsl = np.concatenate([spikecnt_bsl, np.full([nfill, cfg.num_units], np.nan)], axis=0)

            invalid = dgNeuData.session_all.invalid[smpl_idxs]
            invalid = np.concatenate([invalid, np.full([nfill], True)])

            stationary = dgNeuData.session_all.stationary_mask[smpl_idxs]
            stationary = np.concatenate([stationary, np.full([nfill], False)])

            running = dgNeuData.session_all.running_mask[smpl_idxs]
            running = np.concatenate([running, np.full([nfill], False)])

            tgt = np.full([cfg.num_repeats], cid)
            pid = dgNeuData.session_all.stimTable.iloc[smpl_idxs].index.values
            pid = np.concatenate([pid, np.full([nfill], np.nan)])

            start_time = dgNeuData.session_all.stimTable.iloc[smpl_idxs].start_time.values
            stop_time = dgNeuData.session_all.stimTable.iloc[smpl_idxs].stop_time.values
            intervals = np.stack([start_time, stop_time], axis=1)
            intervals = np.concatenate([intervals, np.full([nfill, 2], np.nan)])

            state = dgNeuData.session_all.state_index[smpl_idxs]
            state = np.concatenate([state, np.full([nfill], np.nan)])

            RspData.spike_cnt_ost.append(spikecnt_ost)
            RspData.spike_cnt_stm.append(spikecnt_stm)
            RspData.spike_cnt_pre.append(spikecnt_pre)
            RspData.spike_cnt_bsl.append(spikecnt_bsl)
            RspData.invalid.append(invalid)
            RspData.stationary.append(stationary)
            RspData.running.append(running)
            RspData.intervals.append(intervals)
            RspData.targets.append(tgt)
            RspData.stim_pid.append(pid)
            RspData.state_index.append(state)

        RspData.invalid = np.stack(RspData.invalid, axis=0)
        RspData.stationary = np.stack(RspData.stationary, axis=0)
        RspData.running = np.stack(RspData.running, axis=0)
        RspData.intervals = np.stack(RspData.intervals, axis=0)
        RspData.targets = np.stack(RspData.targets, axis=0)
        RspData.stim_pid = np.stack(RspData.stim_pid, axis=0)
        RspData.state_index = np.stack(RspData.state_index, axis=0)

        spikecnt_ost = np.stack(RspData.spike_cnt_ost, axis=0)
        spikecnt_ost = xr.DataArray(spikecnt_ost, dims=['dg_id', 'repeats', 'unit_id'],
                                            coords={'dg_id': cfg.stim_clsIds,
                                                'repeats': np.arange(cfg.num_repeats),
                                                'unit_id': dgNeuData.session_all.rsp_spikecnt_500ms.unit_id.values})

        spikecnt_stm = np.stack(RspData.spike_cnt_stm, axis=0)
        spikecnt_stm = xr.DataArray(spikecnt_stm, coords=spikecnt_ost.coords)

        spikecnt_pre = np.stack(RspData.spike_cnt_pre, axis=0)
        spikecnt_pre = xr.DataArray(spikecnt_pre, coords=spikecnt_ost.coords)

        spikecnt_bsl = np.stack(RspData.spike_cnt_bsl, axis=0)
        spikecnt_bsl = xr.DataArray(spikecnt_bsl, coords=spikecnt_ost.coords)

        RspData.spike_cnt_ost = spikecnt_ost.transpose('repeats', 'dg_id', 'unit_id')
        RspData.spike_cnt_stm = spikecnt_stm.transpose('repeats', 'dg_id', 'unit_id')
        RspData.spike_cnt_pre = spikecnt_pre.transpose('repeats', 'dg_id', 'unit_id')
        RspData.spike_cnt_bsl = spikecnt_bsl.transpose('repeats', 'dg_id', 'unit_id')

        # pre-stimulus normalization
        spikecnt_mean = np.nanmean(SpikeData.spike_cnt, axis=0)
        spikecnt_std = np.nanstd(SpikeData.spike_cnt, axis=0)
        spikepre_n = RspData.spike_cnt_pre.values.reshape(-1, cfg.num_units) / spikecnt_mean
        spikepre_n = xr.DataArray(spikepre_n.reshape(RspData.spike_cnt_pre.shape), coords=RspData.spike_cnt_pre.coords)

        RspData.spike_cnt_pre_n = spikepre_n

        # response
        RspData.response = RspData.spike_cnt_ost - RspData.spike_cnt_bsl
        # normalization 1
        norm_base = np.fabs(RspData.spike_cnt_ost.values) + np.fabs(RspData.spike_cnt_bsl.values)
        norm_base[norm_base == 0] = np.nan
        response_n = RspData.response.values / norm_base
        RspData.response_n = xr.DataArray(response_n, coords=RspData.response.coords)

        # normalization 2
        norm_base = np.fabs(np.nanmean(RspData.spike_cnt_ost.values, axis=0)) + np.fabs(np.nanmean(RspData.spike_cnt_bsl.values, axis=0))
        norm_base = np.expand_dims(norm_base, axis=0).repeat(cfg.num_repeats, axis=0)
        norm_base[norm_base == 0] = np.nan
        response_n = RspData.response.values / norm_base
        response_n[response_n > 4] = 4
        response_n[response_n < -4] = -4
        RspData.response_n2 = xr.DataArray(response_n, coords=RspData.response.coords)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            spikepre_pos = RspData.spike_cnt_pre_n.sel(unit_id=SessData.units.index[CascadeData.posdly_umsk])
            spikepre_neg = RspData.spike_cnt_pre_n.sel(unit_id=SessData.units.index[CascadeData.negdly_umsk])

            RspData.spkn_pos_mean = np.nanmean(spikepre_pos, axis=2).T
            RspData.spkn_neg_mean = np.nanmean(spikepre_neg, axis=2).T
            RspData.spkn_ratio = RspData.spkn_pos_mean / RspData.spkn_neg_mean
            RspData.spkn_diff = RspData.spkn_pos_mean - RspData.spkn_neg_mean

        # unit stimulus selectivity dataframe
        unitStimSel_df = SessData.units[['ecephys_structure_acronym']].reset_index()

        for ci in cfg.stim_clsIds:
            rsp = RspData.spike_cnt_stm[:,ci,:]
            bsl = RspData.spike_cnt_bsl[:,ci,:]
            invalid = RspData.invalid[ci]

            tscore, pvals = stats.ttest_rel(rsp[~invalid], bsl[~invalid], axis=0)
            unitStimSel_df.loc[:, 'pval_{:d}'.format(ci)] = pvals
            unitStimSel_df.loc[:, 'tscore_{:d}'.format(ci)] = tscore    

        # stimulus selectivity mask
        stim_sel_msk = unitStimSel_df[['pval_{:d}'.format(ci) for ci in cfg.stim_clsIds]].values < cfg.sel_p_thr

        # modulation sign (positive / negative)
        tscores = unitStimSel_df[['tscore_{:d}'.format(ci) for ci in cfg.stim_clsIds]].values
        stim_mod_sign = np.full(tscores.shape, np.nan)
        stim_mod_sign[tscores > 0] = 1
        stim_mod_sign[tscores < 0] = -1

        # non-selectivity mask
        stim_ctl_msk = unitStimSel_df[['pval_{:d}'.format(ci) for ci in cfg.stim_clsIds]].values > cfg.ctl_p_thr

        stim_sel_msk = np.expand_dims(stim_sel_msk.T, axis=0).repeat(cfg.num_repeats, axis=0)
        stim_mod_sign = np.expand_dims(stim_mod_sign.T, axis=0).repeat(cfg.num_repeats, axis=0)
        stim_tscores = np.expand_dims(tscores.T, axis=0).repeat(cfg.num_repeats, axis=0)
        stim_ctl_msk = np.expand_dims(stim_ctl_msk.T, axis=0).repeat(cfg.num_repeats, axis=0)

        spkn_ratio = np.expand_dims(RspData.spkn_ratio.T, axis=2).repeat(cfg.num_units, axis=2)
        spkn_diff = np.expand_dims(RspData.spkn_diff.T, axis=2).repeat(cfg.num_units, axis=2)
        state_index = np.expand_dims(RspData.state_index.T, axis=2).repeat(cfg.num_units, axis=2)
        running = np.expand_dims(RspData.running.T, axis=2).repeat(cfg.num_units, axis=2)
        stationary = np.expand_dims(RspData.stationary.T, axis=2).repeat(cfg.num_units, axis=2)
        invalid = np.expand_dims(RspData.invalid.T, axis=2).repeat(cfg.num_units, axis=2)
        stim_pid = np.expand_dims(RspData.stim_pid.T, axis=2).repeat(cfg.num_units, axis=2)

        RspData.stim_sel_msk = stim_sel_msk
        RspData.stim_mod_sign = stim_mod_sign
        RspData.stim_tscores = stim_tscores
        RspData.stim_ctl_msk = stim_ctl_msk
        RspData.running = running
        RspData.stationary = stationary
        RspData.invalid = invalid
        RspData.spkn_ratio = spkn_ratio
        RspData.spkn_diff = spkn_diff
        RspData.state_index = state_index
        RspData.stim_pid = stim_pid

        unit_ids = np.arange(cfg.num_units).reshape(1,-1)\
                                        .repeat([cfg.num_repeats * len(cfg.stim_clsIds)], axis=0)\
                                        .reshape(stim_sel_msk.shape)
        stim_ids = np.array(cfg.stim_clsIds).reshape(1,-1)\
                                            .repeat(cfg.num_repeats, axis=0)\
                                            .reshape(cfg.num_repeats, -1, 1)\
                                            .repeat(cfg.num_units, axis=2)
        RspData.unit_ids = unit_ids
        RspData.stim_ids = stim_ids

        """ unit responses to preferred stimulus """
        UnitRspData = DataContainer()

        """ all states """
        cond_msks = ~invalid
        CondData = get_response_for_significant_stim_unit_pair(RspData, cfg, cond_msks)
        UnitRspData.selective_rsp = CondData

        CondData = get_response_for_control_stim_unit_pair(RspData, cfg, cond_msks)
        UnitRspData.control_rsp = CondData

        UnitRspData.cfg = cfg

        return UnitRspData

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
        print('the subject has insufficient cascade data, skip ...')
        DgRspData1 = DataContainer()
        save_data_to_file(DgRspData1)
        return

    df = dgNeuData.session_all.stimTable.loc[dgNeuData.session_all.imgStimID][['start_time']]
    df.start_time = df.start_time - 0.5
    df['stop_time'] = df.start_time + 0.5
    tbins = df.values.reshape(-1)

    # state index
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


    DgRspData1 = DataContainer()
    DgRspData = DgRspData1

    """ Drifting-grating stimulus (orientation only) """
    s = 'Drifting-grating stimulus (orientation only)'
    s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
    print(s)

    stim_name2id, clsId2sampleIdx, sample_clsIds = get_DG_classes(dgNeuData, type='ori')
    cfg = DataContainer()

    cfg.num_repeats = 150
    cfg.sel_p_thr = 0.001 # p-value threshold for stimulus-selectivity
    cfg.ctl_p_thr = 0.1 # p-value threshold for control
    cfg.min_sample_per_condition = 2 # minimun samples (trails) to compute average response and SNRs
    cfg.num_samples = dgNeuData.session_all.rsp_spikecnt_200ms.shape[0]
    cfg.num_units = dgNeuData.session_all.rsp_spikecnt_200ms.shape[1]
    cfg.stim_clsIds = sorted(clsId2sampleIdx.keys())[1:]
    cfg.stim_name2id = stim_name2id
    cfg.clsId2sampleIdx = clsId2sampleIdx
    cfg.sample_clsIds = sample_clsIds
    DgRspData.dg_ori = _get_DG_response(cfg)

    """ Drifting-grating stimulus (temporal-frequency only) """
    s = 'Drifting-grating stimulus (temporal-frequency only)'
    s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
    print(s)

    stim_name2id, clsId2sampleIdx, sample_clsIds = get_DG_classes(dgNeuData, type='tfreq')
    cfg = DataContainer()

    cfg.num_repeats = 200
    cfg.sel_p_thr = 0.001 # p-value threshold for stimulus-selectivity
    cfg.ctl_p_thr = 0.3 # p-value threshold for control
    cfg.min_sample_per_condition = 2 # minimun samples (trails) to compute average response and SNRs
    cfg.num_samples = dgNeuData.session_all.rsp_spikecnt_200ms.shape[0]
    cfg.num_units = dgNeuData.session_all.rsp_spikecnt_200ms.shape[1]
    cfg.stim_clsIds = sorted(clsId2sampleIdx.keys())[1:]
    cfg.stim_name2id = stim_name2id
    cfg.clsId2sampleIdx = clsId2sampleIdx
    cfg.sample_clsIds = sample_clsIds
    DgRspData.dg_tfreq = _get_DG_response(cfg)

    """ Drifting-grating stimulus (orientation + temporal-frequency) """
    s = 'Drifting-grating stimulus (orientation + temporal-frequency)'
    s = f'processing [{bcolors.OKBLUE}' + s + f'{bcolors.ENDC}]'
    print(s)

    stim_name2id, clsId2sampleIdx, sample_clsIds = get_DG_classes(dgNeuData, type='both')
    cfg = DataContainer()

    cfg.num_repeats = 100
    cfg.sel_p_thr = 0.001 # p-value threshold for stimulus-selectivity
    cfg.ctl_p_thr = 0.1 # p-value threshold for control
    cfg.min_sample_per_condition = 2 # minimun samples (trails) to compute average response and SNRs
    cfg.num_samples = dgNeuData.session_all.rsp_spikecnt_200ms.shape[0]
    cfg.num_units = dgNeuData.session_all.rsp_spikecnt_200ms.shape[1]
    cfg.stim_clsIds = sorted(clsId2sampleIdx.keys())[1:]
    cfg.stim_name2id = stim_name2id
    cfg.clsId2sampleIdx = clsId2sampleIdx
    cfg.sample_clsIds = sample_clsIds
    DgRspData.dg_both = _get_DG_response(cfg)

    """ save data """
    save_data_to_file(DgRspData1)

def proc_dg_single_neuron_response(resume=False):
    sessions2proc = session_ob_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing single neuron drifting-grating response for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))

        compute_single_neuron_DG_response_index(sess_id, resume=resume)

def main():
    proc_dg_single_neuron_response(resume=False)

if __name__ == "__main__":
    main()