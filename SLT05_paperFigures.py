#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:34:23 2024

@author: 4vt
"""
import os
os.chdir('/home/4vt/Documents/data/SLT05_MSDpostprocess/')
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

fig_dir = 'presentations_and_reports/paper/figures/'
cutoffs = [0.25,0.75]
size = (6,6)
filetype = 'png'
fsize = 12
ptsize = 15
rng = np.random.default_rng(1)

def read_data(path):
    files = ('positive_lipids.tsv', 'negative_lipids.tsv', 'reanalyze_lipids.tsv')
    data = [pd.read_csv(os.path.join(path, file), sep = '\t') for file in files]
    data = pd.concat(data)
    return data
#True positive rate for ROC plots
def TPR(tn,fp,fn,tp):
    divisor = tp + fn
    if divisor == 0:
        return 0
    else:
        return tp/divisor

#False positive rate for ROC plots
def FPR(tn,fp,fn,tp):
    divisor = fp + tn
    if divisor == 0:
        return 0
    else:
        return fp/divisor

#False discovery rate for ROC plots
def FDR(tn,fp,fn,tp):
    divisor = fp + tp
    if divisor == 0:
        return 0
    else:
        return fp/divisor

def ROC(scores, labels):
    #calculate ROC curve
    tprs = [0]
    fprs = [0]
    for cut in sorted(list(scores), reverse=True):
        calls = [yhat >= cut for yhat in scores]
        tn,fp,fn,tp = confusion_matrix(labels, calls).flatten()
        tprs.append(TPR(tn,fp,fn,tp))
        fprs.append(FPR(tn,fp,fn,tp))
    tprs.append(1)
    fprs.append(1)
    return (tprs, fprs)

# =============================================================================
# Figure 2
# =============================================================================
path = 'data/current_datasets/validation_files_with_annotations/QE_Pro_model_validation/'
data = read_data(path)
data = data[np.isfinite(data['label'])]

scores = data['score']
labels = data['label']

tprs, fprs = ROC(scores, labels)

#make base plot
fig, ax = plt.subplots(figsize = size)
ax.plot(fprs,tprs,'-k', linewidth = 1)
ax.plot([0,1],[0,1], '--g', linewidth = 0.5)

#annotate cutoffs
for cutoff in cutoffs:
    calls = [yhat >= cutoff for yhat in scores]
    tn,fp,fn,tp = confusion_matrix(labels, calls).flatten()
    tpr = TPR(tn,fp,fn,tp)
    fpr = FPR(tn,fp,fn,tp)
    ax.scatter(fpr, tpr, s = ptsize, color = 'g', marker = '.')
    fdr = FDR(tn,fp,fn,tp)
    ax.text(fpr, tpr, f'cutoff: {cutoff}\nFDR: {"%.2f"%(fdr)}\nrecall: {"%.2f"%(tpr)}', 
            ha = 'left', va = 'top', fontsize = fsize)

#format plot
y0,y1 = ax.get_ylim()
x0,x1 = ax.get_xlim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))
ax.set_ylim(-.001,1.001)
ax.set_xlim(-.001,1.001)
# ax.set_facecolor('lightgrey')
ax.set_ylabel('True Positive Rate', fontsize = fsize)
ax.set_xlabel('False Positive Rate', fontsize = fsize)
aucroc = roc_auc_score(labels, scores)
ax.annotate(f'AUC: {"%.2f"%(aucroc)}', (0.5,0.5), ha='left', va='top', fontsize = fsize)
# ax.set_title('B', loc = 'left')
fig.savefig(os.path.join(fig_dir, f'Figure2.{filetype}'), bbox_inches = 'tight', dpi = 900)
plt.close('all')

# =============================================================================
# Figure 3
# =============================================================================
instruments = ['LTQPro', 'QE', 'QTOF']

all_data = {}
for trained in instruments:
    for tested in instruments:
        path = f'data/other_analyses/instrumentCrossTraining/instrumentCrossTraining_{trained}|{tested}/'
        all_data[(trained, tested)] = read_data(path)
        

bins = np.linspace(0,1,50)
yheight = 0
fig, axes = plt.subplots(nrows = 3, ncols = 3, sharex = True, #sharey = True,
                         layout = 'constrained', figsize = (8,6))
for i, trained in enumerate(instruments):
    for j, tested in enumerate(reversed(instruments)):
        data = all_data[(trained, tested)]
        assert min(data['score']) >= 0
        assert max(data['score']) <= 1

        ax = axes[i,j]
        if i == 2:
            ax.set_xlabel('Score')
        else:
            ax.xaxis.set_ticks_position('none') 
        if i == 0:
            ax.set_title(tested, fontsize = 10)
        if j == 0:
            ax.set_ylabel('Number of Lipids')
        else:
            ax.yaxis.set_ticks_position('none') 
        if j == 2:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(trained)
        ax.tick_params(axis='both', which='major', labelsize=7)    
        
        if 'split' in data.columns:
            data = data[data['split'] == 'test']
        ax.hist(data[data['label'] == 1]['score'],
                bins = bins, 
                color = 'g', 
                alpha = 0.5, 
                label = 'True',
                histtype='stepfilled')
        ax.hist(data[data['label'] == 0]['score'], 
                bins = bins, 
                color = 'k', 
                alpha = 0.5, 
                label = 'False',
                histtype='stepfilled')
        
        y0,y1 = ax.get_ylim()
        for cut in cutoffs:
            ax.plot([cut]*2, (0,y1), '-k', linewidth = 1)
        ax.set_ylim(0,y1)
        # yheight = max(y1, yheight)
        
        if i == 0 and j == 2:
            # black_patch = mpatches.Patch(color='k', label='Cutoff')
            ax.legend(loc = 'right', bbox_to_anchor = (1.72, 0.5))
    ax.set_xlim(0,1)

fig.supxlabel('Instrument Used for Testing', x = 0.5, y = 1)
fig.supylabel('Instrument Used for Training', x = 0.88, y = 0.5)

fig.savefig(os.path.join(fig_dir, f'Figure3.{filetype}'), bbox_inches = 'tight', dpi = 900)
plt.close('all')

# =============================================================================
# Figure 4
# =============================================================================

classes = [f'data/other_analyses/LOOclassCV/{f}' for f in os.listdir('data/other_analyses/LOOclassCV')]

class_data = {}
for holdout in classes:
    ontology = re.search(r'/LOOclassCV_(.+)\Z', holdout).group(1)
    data = read_data(holdout)
    data = data[data['split'] == 'test']
    data = data[np.isfinite(data['label'])]
    if np.sum(data['label']) > 20 and len(set(data['label'])) == 2:
        class_data[ontology] = data

def get_colors(vals):
    low = min(vals)
    high = max(vals)
    return [cm.viridis(int(((val-low)/(high-low))*cm.viridis.N)) for val in vals]


def get_sm(vals):
    colormap = matplotlib.colormaps['viridis']
    sm = plt.cm.ScalarMappable(cmap=colormap)
    sm.set_clim(vmin = min(vals), vmax = max(vals))
    return sm

colorvals = [d.shape[0] for d in class_data.values()]
colors = get_colors(colorvals)
sm = get_sm(colorvals)

fig, ax = plt.subplots()
for i,lclass in enumerate(class_data.keys()):
    data = class_data[lclass]

    tprs, fprs = ROC(data['score'], data['label'])
    ax.plot(fprs, tprs, '-', color = colors[i], linewidth = 1)

fsize = 8
ax.text(0.215, .52, 'PI', ha = 'left', va = 'top', fontsize = fsize)
ax.text(0.47, .88, 'PG', ha = 'left', va = 'top', fontsize = fsize)

clb = fig.colorbar(sm, ax = ax, location = 'right', shrink = 0.825)
clb.set_label('Number of Lipids')

ax.plot([0,1],[0,1], '--g', linewidth = 0.5)
y0,y1 = ax.get_ylim()
x0,x1 = ax.get_xlim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))
ax.set_ylim(-.001,1.001)
ax.set_xlim(-.001,1.001)
# ax.set_facecolor('lightgrey')
ax.set_ylabel('True Positive Rate', fontsize = fsize)
ax.set_xlabel('False Positive Rate', fontsize = fsize)

fig.savefig(os.path.join(fig_dir, f'Figure4.{filetype}'), bbox_inches = 'tight', dpi = 900)
plt.close('all')

# =============================================================================
# Figure S1
# =============================================================================

data_dir = 'MSDpostprocess/build/QE_Pro_model_training/'
data_files = [f for f in os.listdir(data_dir) if f.endswith('.tsv') and not f.startswith('not_')]
good = []
for file in data_files:
    data = pd.read_csv(f'{data_dir}{file}', sep = '\t')
    data = data[data['label'] == 1]
    good.append(data)
good = pd.concat(good)
predictors = ['Dot product', 'S/N average', 'isotope_error', 'mz_error', 'rt_error']
tolog = ['S/N average', 'isotope_error']
good[tolog] = np.log(good[tolog])

plsda_model = Pipeline([('scalar', StandardScaler()),
                        ('logreg', PLSRegression(n_components=2))])
lreg_model = Pipeline([('scalar', StandardScaler()),
                       ('logreg', LogisticRegression())])

highlight = 'g'
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (8,8),
                       layout = 'constrained')
for i,lipid in enumerate(['PI', 'PG']):
    y = good['Ontology'] == lipid
    features = good[predictors].copy()
    
    #PLS-DA
    X, _ = plsda_model.fit_transform(X = features, y = np.array(y, np.logical_not(y)).T)
    ax = axes[0,i]
    ax.scatter(X[:,0], X[:,1], s = 20, marker = '.', alpha = 0.5,
               c = [highlight if o else 'k' for o in y])
    ax.set_title(lipid)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    if i == 1:
        ax2 = ax.twinx()
        ax2.set_ylabel('PLS-DA')
        ax2.set_yticks([])

    
    #Logistic Regression
    tofold = ['mz_error', 'rt_error']
    features[tofold] = np.abs(features[tofold])
    
    lreg = lreg_model.fit(X = features, 
                          y = y)
    ŷ = lreg.predict_proba(X = features)
    ax = axes[1,i]
    Δ = 0.2
    ax.scatter(y + rng.uniform(-Δ,Δ,len(y)), ŷ[:,1], alpha = 0.5, 
               s = 20, marker = '.', c = [highlight if o else 'k' for o in y])
    ax.set_xticks((0,1), ('other lipids', lipid))
    ax.set_ylabel(f'{lipid} Probability')
    ax.set_ylim(0,1)
    if i == 1:
        ax2 = ax.twinx()
        ax2.set_ylabel('Logistic Regression')
        ax2.set_yticks([])
    else:
        patches = [mpatches.Patch(color='k', label='other lipids'),
                   mpatches.Patch(color=highlight, label='class of interest')]
        ax.legend(handles = patches)
fig.savefig(f'{fig_dir}FigureS1.png',
            bbox_inches = 'tight', dpi = 900)
