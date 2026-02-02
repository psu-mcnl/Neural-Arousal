import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
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

mcc = MouseConnectivityCache(resolution = 25, 
                             manifest_file = settings.connectivity.manifest_path)
structure_tree = mcc.get_structure_tree()

name_map = structure_tree.get_name_map() # dictionary mapping ids to structure names
acrnm_map = structure_tree.get_id_acronym_map() # dictionary mapping acronyms to ids
colormap = structure_tree.get_colormap() # the colormap used for the allen 3D mouse atlas ontology
id_map = structure_tree.value_map(lambda x: x['id'], lambda y: y['acronym'])

template, template_info = mcc.get_template_volume()
annot, annot_info = mcc.get_annotation_volume()

import pyvista as pv

pv.start_xvfb()

cpos = [(-381.74, -46.02, 216.54), (74.8305, 89.2905, 100.0), (0.23, 0.072, 0.97)]
p = pv.Plotter()
p.add_volume(template, cmap="bone", opacity="sigmoid")
p.camera_position = cpos

p.save_graphic('brain_3d.pdf')