#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:38:03 2024

@author: 4vt
"""

import os
import re
from collections import defaultdict
from collections import Counter

from sortedcontainers import SortedList
import pandas as pd
import numpy as np

os.chdir('/home/4vt/Documents/data/SLT05_LipoCLEAN/data/current_datasets/MSD5/')

file_mapping =  [('Aspergillus37_neg_mz_MD5.txt', 'MSD4_LTQPro_LCO-Asp37_mzexport_negative_8-23-23.txt'),
                 ('Aspergillus37_pos_mz_MD5.txt', 'MSD4_LTQPro_LCO-Asp37_mzexport_positive_8-23-23.txt'),
                 ('Laccaria_neg_mz_MD5.txt', 'MSD4_LTQPro_LCO-Laccaria_mzexport_negative.txt'),
                 ('Laccaria_pos_mz_MD5.txt', 'MSD4_LTQPro_LCO-Laccaria_mzexport_positive.txt'),
                 ('QE_neg_mz_MD5.txt', 'MSD4_QE_MTBLS5583_mzexport_negative_andrewcpu.txt'),
                 ('QE_pos_mz_MD5.txt', 'MSD4_QE_MTBLS5583_mzexport_positive_0_202456119.txt'),
                 ('QTOF_neg_mz_MD5.txt', 'MSD4_QTOF_MTBLS4108_mzexport_negative_firstproc.txt'),
                 ('QTOF_pos_mz_MD5.txt', 'MSD4_QTOF_MTBLS4108_mzexport_positive_mzexport.txt')]

#this finds features in the new data that are close to labeled features in the old data
def find_matches(mz, rt, idx):
    mz_matches = [m[1] for m in old_mz_list.irange((mz - Δmz,), (mz + Δmz,))]
    rt_matches = [(m[1], np.abs(m[0] - rt)) for m in old_rt_list.irange((rt - Δrt,), (rt + Δrt,))]
    match_indices = set(mz_matches).intersection([m[0] for m in rt_matches])
    matches = [m for m in rt_matches if m[0] in match_indices]
    return (idx, matches)

#a hit is bad if there are multiple features in the old data and they do not all have the same
#lipid_name and label
isbad = lambda h: len(set(old_data.loc[[m[0] for m in h[1]],'name_label'])) > 1

#labels do not get transfered if isbad is true
#or if the feature was labeled as a false positive but MS-DIAL 5 identifies it differently from 4
#this is because we cannot be sure that the new identification is also false.
def transfer_label(hit):
    if isbad(hit):
        return np.nan
    hit_subset = old_data.loc[[h[0] for h in hit[1]]]
    hit_lipid = next(l for l in hit_subset['lipid_name'])
    hit_label = next(l for l in hit_subset['label'])
    query_lipid = new_data.loc[hit[0], 'lipid_name']
    if query_lipid == hit_lipid:
        return hit_label
    else:
        if hit_label == 1:
            return 0
        elif hit_label == 0:
            uncertain_mappings.append(hit)
            return np.nan
    raise NotImplementedError

summary_stats = []
for new_filename, old_filename in file_mapping:
    #the raw mzmatrix export from MS-DIAL 5
    new_data = pd.read_csv(new_filename, 
                           sep = '\t',
                           skiprows = 4)
    new_data = new_data[new_data['MS/MS matched']]
    
    #the labeled training data searched with MS-DIAL 4
    old_data = pd.read_csv(old_filename,
                           sep = '\t',
                           skiprows = 4)
    old_data = old_data[np.isfinite(old_data['label'])]
    old_data = old_data[[c for c in old_data.columns if not c.startswith('Unnamed:')]]

    #we only care about summed tails so we drop the individual tail calls for comparison
    new_data['lipid_name'] = [re.sub(r'(?:\||/).+\Z', '', name) for name in new_data['Metabolite name']]
    new_data['lipid_name'] = [re.sub(r'low score: ', '', name) for name in new_data['lipid_name']]
    old_data['lipid_name'] = [re.sub(r'(?:\||/).+\Z', '', name) for name in old_data['Metabolite name']]
    
    #set up data structures for fast lookup of relavent data
    new_id_align = {i:a for i,a in zip(new_data.index, new_data['Alignment ID'])}
    old_id_align = {i:a for i,a in zip(old_data.index, old_data['Alignment ID'])}
    old_mz_list = SortedList(zip(old_data['Average Mz'], old_data.index))
    old_rt_list = SortedList(zip(old_data['Average Rt(min)'], old_data.index))
    
    #set the m/z tolerance based on the mean standard deviation of m/z values
    all_stdevs = []
    for data in [old_data, new_data]:
        qcol_start = list(data.columns).index('MS/MS spectrum') + 1
        quant_cols = [c for c in data.columns[qcol_start:] if c != 'lipid_name']
        stdevs = np.nanstd(data[quant_cols], axis = 1)
        all_stdevs.extend(stdevs)
    Δmz = np.nanmean(all_stdevs)*2
    
    #collect potential matches with a wide Δrt value
    Δrt = 5
    hits = [find_matches(mz, rt, i) for mz, rt, i in zip(new_data['Average Mz'], new_data['Average Rt(min)'], new_data.index)]
    hits = [h for h in hits if len(h[1]) > 0]

    #find a retention time cutoff that controls the number of incoherent multi-matches at a reasonable level
    old_data['name_label'] = [f'{n}{l}' for n,l in zip(old_data['lipid_name'], old_data['label'])]
    cutoffs = np.linspace(Δrt, 0.1, 100)
    n_hits = len([h for h in hits if len(h[1]) > 0])
    for cutoff in cutoffs:
        filtered = []
        for hit in hits:
            hit = [hit[0], [h for h in hit[1] if h[1] < cutoff]]
            if len(hit[1]) > 0:
                filtered.append(hit)
        hits = filtered
        multihits = [h for h in hits if len(h[1]) > 1]
        n_bad = np.sum([isbad(h) for h in multihits])
        if n_bad/n_hits < 0.05:
            break
       
    #transfer labels to MS-DIAL 5 data
    uncertain_mappings = []
    transferred_labels = {h[0]:transfer_label(h) for h in hits}
    transferred_labels = defaultdict(lambda: np.nan, transferred_labels)
    new_data['label'] = [transferred_labels[i] for i in new_data.index]
    
    #save newly labeled data
    new_data.to_csv(f'labeled_MSD5_data/{new_filename}')
    
    
    #collect data for the uncertianly mapped population
    unc_new_old_idx_map = {m[0]:[h[0] for h in m[1]] for m in uncertain_mappings}
    unc_ids = list(unc_new_old_idx_map.keys())

    unc_old_new_idx_map = defaultdict(lambda:[])
    for idx in unc_ids:
        for hit in unc_new_old_idx_map[idx]:
            unc_old_new_idx_map[hit].append(idx)

    unc_new = new_data.loc[unc_ids]
    unc_new['hits'] = [';'.join(str(old_id_align[i]) for i in unc_new_old_idx_map[h]) for h in unc_ids]
    unc_new.to_csv(f'uncertain_mappings/{new_filename}',
                   sep = '\t', 
                   index = False)
    
    unc_hit_ids = unc_old_new_idx_map.keys()
    unc_old = old_data.loc[unc_hit_ids]
    unc_old['hits'] = [';'.join(str(new_id_align[i]) for i in unc_old_new_idx_map[h]) for h in unc_hit_ids]
    unc_new.to_csv(f'uncertain_mappings/{old_filename}',
                   sep = '\t', 
                   index = False)
    
    #store information about the file
    new_label_counts = Counter(new_data['label'])
    old_label_counts = Counter(old_data['label'])
    summary_stats.append((new_filename, 
                          old_filename, 
                          n_hits,
                          n_bad,
                          cutoff, 
                          Δmz,
                          new_label_counts[0],
                          new_label_counts[1],
                          new_label_counts[0] + new_label_counts[1],
                          new_data.shape[0],
                          old_label_counts[0],
                          old_label_counts[1],
                          old_label_counts[0] + old_label_counts[1],
                          old_data.shape[0],
                          len(uncertain_mappings)))

#save summary statistics
summary_stats = pd.DataFrame(summary_stats)
summary_stats.columns = ('new_filename', 
                         'old_filename', 
                         'N_hits',
                         'N_incoherent_multihits',
                         'RT_cutoff', 
                         'Δmz',
                         'N_labeled_0_new',
                         'N_labeled_1_new',
                         'N_labeled_total_new',
                         'N_total_new',
                         'N_labeled_0_old',
                         'N_labeled_1_old',
                         'N_labeled_total_old',
                         'N_total_old',
                         'N_uncertain')
summary_stats.to_csv('label_transfer_summary_statisitics.tsv', sep = '\t', index = False)
