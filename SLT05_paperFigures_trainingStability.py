#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:59:26 2024

@author: 4vt
"""
import sys
from multiprocessing import Pool

from MSDpostprocess.options import options, setup_workspace, validate_inputs
from MSDpostprocess.utilities import read_files, filter_data, write_data, split_index
from MSDpostprocess.models import mz_correction, rt_correction, predictor_model, add_isotope_error

def run_MSDpostprocess(seed):
    args = options()
    args.output = f'trainingStability_{seed}'
    args.model = f'model_trainingStability_{seed}'
    args.seed = seed
    
    validate_inputs(args)
    setup_workspace(args)
    
    lipid_data = read_files(args)
    lipid_data = filter_data(lipid_data, args)
    lipid_data = add_isotope_error(lipid_data)
    
    mz_model = mz_correction(args)
    rt_model = rt_correction(args)
    final_model = predictor_model(args)   
    
    train_idx, test_idx = split_index(lipid_data, args)
    lipid_data['split'] = ['train' if i in train_idx else 'test' for i in lipid_data.index]
    
    mz_model.fit(lipid_data.loc[train_idx])
    lipid_data = mz_model.correct_data(lipid_data)

    rt_model.fit(lipid_data.loc[train_idx])
    lipid_data = rt_model.correct_data(lipid_data)
    
    final_model.fit(lipid_data.loc[train_idx])

    lipid_data = final_model.classify(lipid_data)
    write_data(lipid_data, args)

N = 32
jobs = range(N)
sys.argv += '--options options_trainingStability.toml'.split()
with Pool(N) as p:
    _ = p.map(run_MSDpostprocess, jobs)
