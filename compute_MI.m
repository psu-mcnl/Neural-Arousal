clc; clear all;
%% script setup
% change the current folder to the folder of this m-file.
currentFolder = fileparts(which(mfilename));
if isempty(currentFolder)
    currentFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
if (~strcmp(currentFolder, pwd))
  cd(currentFolder);
end
addpath(genpath('.'));

% add mutual information toolbox
addpath('/home/yzy161/Dropbox/Code/Library/Matlab/Neuroscience-Information-Theory-Toolbox');


%% general settings
PROJECT_DIR = '~/Data/Project-Data/Neuroscience/Neural-Arousal';
GENERAL_DIR = fullfile(PROJECT_DIR, 'general');
VIS_EPHY_SESS_DIR = fullfile(PROJECT_DIR, 'sessions');

%% prepare for MI computation
sessAllDbs = load(fullfile(GENERAL_DIR, 'sessions_ephys.mat'));

cfg = struct;
cfg.DB_SESS_DIR = VIS_EPHY_SESS_DIR;
cfg.DB_SESS_DIR_TMPL = 'session_%d';
cfg.DB_SPKEVENT_FILE_NAME = 'data4MI_computation.mat';

regions = {'CA1', 'SUB', 'DG', 'Pros', 'CA3', ...
           'VISpm', 'VISp', 'VISl', 'VISal', 'VISrl', 'VISam', ...
           'VISli', 'VISmma', 'VISmmp', ...
           'Eth', 'LP', 'SGN', 'PO', 'LGd', 'VPM', 'MGd', 'MGv', 'POL', 'LGv', ...
           'LD', 'IntG', 'IGL', 'APN', 'NOT'};

regionPairs = {};
for i = 1:length(regions)
    for j = i+1:length(regions) regionPairs{end+1} = {regions{i}, regions{j}}; end
end

%% MI compuation for all sessions (visEhpys)
cfg.SESS_NAMES = sessAllDbs.ob;
cfg.COMPUTE_PVAL = false;

MIResults = struct;
for i = 1:length(regionPairs)
    region1 = regionPairs{i}{1};
    region2 = regionPairs{i}{2};
    fieldName = sprintf('%s_%s', region1, region2);
    
    s = sprintf('[%d|%d] computing mutual information for pair (%s, %s)', ...
                i, length(regionPairs), region1, region2);
    disp(s)
    
    [InfoResults, pVals, nTrails] = func_pairMI(cfg, regionPairs{i});
    
    results = struct('MI', InfoResults, 'pVals', pVals, 'nTrails', nTrails);
    MIResults = setfield(MIResults, fieldName, results);
end

save(fullfile(GENERAL_DIR, 'ob_MI_results.mat'), 'MIResults')

%% MI compuation for all sessions (visEhpys)
cfg.SESS_NAMES = sessAllDbs.fc;
cfg.COMPUTE_PVAL = false;

MIResults = struct;
for i = 1:length(regionPairs)
    region1 = regionPairs{i}{1};
    region2 = regionPairs{i}{2};
    fieldName = sprintf('%s_%s', region1, region2);
    
    s = sprintf('[%d|%d] computing mutual information for pair (%s, %s)', ...
                i, length(regionPairs), region1, region2);
    disp(s)
    
    [InfoResults, pVals, nTrails] = func_pairMI(cfg, regionPairs{i});
    
    results = struct('MI', InfoResults, 'pVals', pVals, 'nTrails', nTrails);
    MIResults = setfield(MIResults, fieldName, results);
end

save(fullfile(GENERAL_DIR, 'fc_MI_results.mat'), 'MIResults')