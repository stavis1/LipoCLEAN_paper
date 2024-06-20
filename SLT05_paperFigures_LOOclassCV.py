#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:29:25 2024

@author: 4vt
"""
import sys
from multiprocessing import Pool

import numpy as np

from MSDpostprocess.options import options, setup_workspace, validate_inputs
from MSDpostprocess.utilities import read_files, filter_data, write_data
from MSDpostprocess.models import mz_correction, rt_correction, predictor_model, add_isotope_error

def run_MSDpostprocess(drop):
    args = options()
    args.output = f'LOOclassCV_{drop}'
    args.model = f'model_LOOclassCV_{drop}'
    
    validate_inputs(args)
    setup_workspace(args)
    
    lipid_data = read_files(args)
    lipid_data = filter_data(lipid_data, args)
    lipid_data = add_isotope_error(lipid_data)
    
    mz_model = mz_correction(args)
    rt_model = rt_correction(args)
    final_model = predictor_model(args)   
    
    # test_idx = lipid_data[lipid_data['Ontology'] == drop].index
    train_idx = lipid_data[lipid_data['Ontology'] != drop].index
    lipid_data['split'] = ['train' if i in train_idx else 'test' for i in lipid_data.index]
    
    mz_model.fit(lipid_data.loc[train_idx])
    lipid_data = mz_model.correct_data(lipid_data)

    rt_model.fit(lipid_data.loc[train_idx])
    lipid_data = rt_model.correct_data(lipid_data)
    
    final_model.fit(lipid_data.loc[train_idx])

    lipid_data = final_model.classify(lipid_data)
    write_data(lipid_data, args)


sys.argv += '--options options_LOOclassCV.toml'.split()
args = options()
lipid_data = read_files(args)
lipid_data = lipid_data[np.isfinite(lipid_data['label'])]

jobs = list(set(lipid_data['Ontology']))

with Pool(args.cores) as p:
    _ = p.map(run_MSDpostprocess, jobs)
