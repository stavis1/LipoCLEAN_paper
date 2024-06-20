#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:17:30 2024

@author: 4vt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:29:25 2024

@author: 4vt
"""
import sys
import os
from multiprocessing import Pool

import numpy as np

from MSDpostprocess.options import options, setup_workspace, validate_inputs
from MSDpostprocess.utilities import read_files, filter_data, write_data, split_index
from MSDpostprocess.models import mz_correction, rt_correction, predictor_model, add_isotope_error
from MSDpostprocess.QC import plot_mz_QC, plot_rt_QC, plot_final_QC, plot_pairwise_scores

def run_MSDpostprocess(train, test):
    args = options()
    args.output = f'instrumentCrossTraining_{train}|{test}'
    args.model = f'model_instrumentCrossTraining_{train}|{test}'
    
    validate_inputs(args)
    setup_workspace(args)
    
    lipid_data = read_files(args)
    lipid_data = filter_data(lipid_data, args)
    lipid_data = lipid_data[[f.startswith(train) or f.startswith(test) for f in lipid_data['file']]]
    lipid_data = add_isotope_error(lipid_data)
    
    mz_model = mz_correction(args)
    rt_model = rt_correction(args)
    final_model = predictor_model(args)   
    
    if train != test:
        test_idx = lipid_data[[f.startswith(test) for f in lipid_data['file']]].index
        train_idx = lipid_data[[f.startswith(train) for f in lipid_data['file']]].index
    else:
        lipid_data = lipid_data[[f.startswith(train) for f in lipid_data['file']]]
        train_idx, test_idx = split_index(lipid_data, args)
        
    lipid_data['split'] = ['train' if i in train_idx else 'test' for i in lipid_data.index]
    
    mz_model.fit(lipid_data.loc[train_idx])
    lipid_data = mz_model.correct_data(lipid_data)

    rt_model.fit(lipid_data.loc[train_idx])
    lipid_data = rt_model.correct_data(lipid_data)
    
    final_model.fit(lipid_data.loc[train_idx])

    lipid_data = final_model.classify(lipid_data)
    write_data(lipid_data, args)

sys.argv += '--options options_instrumentCrossTraining.toml'.split()
args = options()
instruments = ['LTQPro', 'QE', 'QTOF']
jobs = [(tr,te) for tr in instruments for te in instruments]

with Pool(args.cores) as p:
    _ = p.starmap(run_MSDpostprocess, jobs)
