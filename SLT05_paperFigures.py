#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:34:23 2024

@author: 4vt
"""
import os
os.chdir('/home/4vt/Documents/data/SLT05_LipoCLEAN/')
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib
from matplotlib.gridspec import GridSpec
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pyvips

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Verdana'

fig_dir = 'presentations_and_reports/paper/figures/'
cutoffs = [0.25,0.75]
size = np.array((6,6))
filetypes = ['tif', 'png']
fsize = 12
rng = np.random.default_rng(1)
DPI = 900
highlight = 'g'

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

plsda_model = Pipeline([('scalar', StandardScaler()),
                        ('pls', PLSRegression(n_components=2))])
lreg_model = Pipeline([('scalar', StandardScaler()),
                       ('logreg', LogisticRegression())])

# =============================================================================
# Figure 1
# =============================================================================

fig1 = pyvips.Image.new_from_file(fig_dir + 'Figure1.svg', dpi = DPI)
for filetype in filetypes:
    fig1.write_to_file(fig_dir + f'Figure1.{filetype}')

# =============================================================================
# Figure 2
# =============================================================================
instruments = ['OrbiPro', 'QE', 'QTOF']

all_data = {}
for trained in instruments:
    for tested in instruments:
        tr = 'LTQPro' if trained == 'OrbiPro' else trained
        te = 'LTQPro' if tested == 'OrbiPro' else tested
        path = f'data/other_analyses/instrumentCrossTraining/instrumentCrossTraining_{tr}|{te}/'
        all_data[(trained, tested)] = read_data(path)
        

bins = np.linspace(0,1,50)
yheight = 0
scale = 0.82
fig, axes = plt.subplots(nrows = 3, ncols = 3, sharex = True, #sharey = True,
                         layout = 'constrained', figsize = (6*scale,6.5*scale))
for i, trained in enumerate(instruments):
    for j, tested in enumerate(reversed(instruments)):
        data = all_data[(trained, tested)]
        data = data[np.isfinite(data['label'])]
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
            ax.set_ylabel('Frequency of Lipids')
        else:
            ax.yaxis.set_ticks_position('none') 
        if j == 2:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(trained)
        ax.tick_params(axis='both', which='major', labelsize=7) 
        ax.set_yticks([])
        
        if 'split' in data.columns:
            data = data[data['split'] == 'test']
        ax.hist(data[data['label'] == 1]['score'],
                bins = bins, 
                color = highlight, 
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
        
        k = i*3 + j
        ax.text(0.5,y1*0.9,'ABCDEFGHI'[k])
        ax.text(0.5,y1*0.55,f'Nt = {data[data["label"] == 1].shape[0]}',
                ha = 'center', va = 'center', color = highlight)
        ax.text(0.5,y1*0.45,f'Nf = {data[data["label"] == 0].shape[0]}',
                ha = 'center', va = 'center', color = 'k')
        
        ax.set_ylim(0,y1)
        ax.set_xticks((0, 0.25, 0.75, 1),('0', '0.25', '0.75', '1'))
        
        tsize = 9
        ax.tick_params(axis='x', labelsize=tsize)
        ax.tick_params(axis='y', labelsize=tsize)
        
        # if i == 0 and j == 2:
        #     ax.legend(loc = 'upper center', bbox_to_anchor = (0.6,1.54))
    ax.set_xlim(0,1)

fig.supxlabel('Instrument Used for Testing', x = 0.5, y = 1)#y = 0.93)
fig.supylabel('Instrument Used for Training', x = 1, y = 0.5)


for filetype in filetypes:
    fig.savefig(os.path.join(fig_dir, f'Figure2.{filetype}'), bbox_inches = 'tight', dpi = DPI)

# =============================================================================
# Figure 3
# =============================================================================
path = 'data/current_datasets/validation_files_with_annotations/QE_Pro_model_validation/'
data = read_data(path)
data = data[np.isfinite(data['label'])]

scores = data['score']
labels = data['label']

tprs, fprs = ROC(scores, labels)

scale = 0.9
#make base plot
fig, ax = plt.subplots(figsize = size*scale)
ax.plot(fprs,tprs,'-k', linewidth = 1, zorder = -1)
ax.plot([0,1],[0,1], '--g', linewidth = 0.5, zorder = -1)

#annotate cutoffs
for cutoff in cutoffs:
    calls = [yhat >= cutoff for yhat in scores]
    tn,fp,fn,tp = confusion_matrix(labels, calls).flatten()
    tpr = TPR(tn,fp,fn,tp)
    fpr = FPR(tn,fp,fn,tp)
    ax.scatter(fpr, tpr, s = 25, color = highlight, marker = '.', zorder = 1)
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
for filetype in filetypes:
    fig.savefig(os.path.join(fig_dir, f'Figure3.{filetype}'), bbox_inches = 'tight', dpi = DPI)


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

scale = 0.85
fig, ax = plt.subplots(figsize = size*scale)
for i,lclass in enumerate(class_data.keys()):
    data = class_data[lclass]

    tprs, fprs = ROC(data['score'], data['label'])
    ax.plot(fprs, tprs, '-', color = colors[i], linewidth = 1)

fsize = 10
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

for filetype in filetypes:
    fig.savefig(os.path.join(fig_dir, f'Figure4.{filetype}'), bbox_inches = 'tight', dpi = DPI)

# =============================================================================
# Figure S6
# =============================================================================

data_dir = 'data/other_analyses/trainingStability/'
results = []
for result in os.listdir(data_dir):
    data = read_data(data_dir + result)
    data = data[np.isfinite(data['label'])]
    data = data[data['split'] == 'test']
    results.append(data)

fig, ax = plt.subplots()
for c,result in zip(colors,results):
    tprs, fprs = ROC(result['score'], result['label'])
    ax.plot(fprs, tprs, '-k', linewidth = 1, alpha = 0.5)

ax.plot([0,1],[0,1], '--g', linewidth = 0.5)
y0,y1 = ax.get_ylim()
x0,x1 = ax.get_xlim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))
ax.set_ylim(-.001,1.001)
ax.set_xlim(-.001,1.001)
# ax.set_facecolor('lightgrey')
ax.set_ylabel('True Positive Rate', fontsize = fsize)
ax.set_xlabel('False Positive Rate', fontsize = fsize)
for filetype in filetypes:
    fig.savefig(f'{fig_dir}FigureS6.{filetype}',
            bbox_inches = 'tight', dpi = DPI)

# =============================================================================
# table S1
# =============================================================================

data_dir = 'LipoCLEAN/build/build_data/'
files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
data = []
for f in files:
    tmp = pd.read_csv(data_dir + f, sep = '\t', skiprows = 4)
    tmp = tmp[np.isfinite(tmp['label'])]
    dataset = re.search(r'\A([^_]+)_', f).group(1)
    if dataset == 'LTQPro':
        dataset = 1 if 'Laccaria' in f else 2
    else:
        dataset = 3 if dataset == 'QE' else 4
    tmp['dataset'] = [f'Dataset {dataset}']*tmp.shape[0]
    data.append(tmp)
data = pd.concat(data)

class_counts = pd.crosstab(data['Ontology'], data['dataset'])
class_counts['Total'] = np.sum(class_counts, axis =1)
class_counts = class_counts.sort_values('Total', ascending = False)
class_counts.index = [c.replace('_', r'\_') for c in class_counts.index]
class_counts.to_csv('presentations_and_reports/paper/figures/supplementary_table_1.csv')

# =============================================================================
# Figure S7
# =============================================================================

data_dir = 'LipoCLEAN/build/QE_Pro_model_training/'
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
    y0,y1 = ax.get_ylim()
    x0,x1 = ax.get_xlim()
    x_off = .05
    y_off = .92
    ax.text(x0 + (x1-x0)*x_off,
            y0 + (y1-y0)*y_off,
            'AB'[i])
    
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
    y0,y1 = ax.get_ylim()
    x0,x1 = ax.get_xlim()
    ax.text(x0 + (x1-x0)*x_off,
            y0 + (y1-y0)*y_off,
            'CD'[i])
    if i == 1:
        ax2 = ax.twinx()
        ax2.set_ylabel('Logistic Regression')
        ax2.set_yticks([])
    else:
        patches = [mpatches.Patch(color='k', label='other lipids'),
                   mpatches.Patch(color=highlight, label='class of interest')]
        ax.legend(handles = patches)
for filetype in filetypes:
    fig.savefig(f'{fig_dir}FigureS7.{filetype}',
            bbox_inches = 'tight', dpi = DPI)

# =============================================================================
# Figure S4
# =============================================================================

def plot_separability(good, group0, group1):
    predictors = ['Dot product', 'S/N average', 'isotope_error', 'mz_error', 'rt_error']
    tolog = ['S/N average', 'isotope_error']
    good[tolog] = np.log(good[tolog])
    features = good[predictors].copy()
    tofold = ['mz_error', 'rt_error']
    features[tofold] = np.abs(features[tofold])
    y = good['group']
    
    ny, nx = 4,3
    scale = 2
    fig = plt.figure(layout = 'constrained',
                     figsize = (scale*ny,scale*nx))
    gs = GridSpec(nx, ny, figure=fig)
    ax1 = fig.add_subplot(gs[:2,:2])
    ax2 = fig.add_subplot(gs[:2,2:])
    ax3 = fig.add_subplot(gs[2:,1:3])
    
    #PLS-DA
    X, _ = plsda_model.fit_transform(X = features, y = np.array(y, np.logical_not(y)).T)
    fit_plsda = plsda_model.fit(X = features, y = np.array(y, np.logical_not(y)).T)
    ax1.scatter(X[:,0], X[:,1], s = 20, marker = '.', alpha = 0.5,
               c = [highlight if o else 'k' for o in y])
    ax1.set_title('A', loc = 'right')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    patches = [mpatches.Patch(color='k', label=f'{group0} Data'),
               mpatches.Patch(color=highlight, label=f'{group1} Data')]
    ax1.legend(handles = patches)
    
    #Logistic Regression
    lreg = lreg_model.fit(X = features, 
                          y = y)
    ŷ = lreg.predict_proba(X = features)
    Δ = 0.2
    ax2.scatter(y + rng.uniform(-Δ,Δ,len(y)), ŷ[:,1], alpha = 0.5, 
               s = 20, marker = '.', c = [highlight if o else 'k' for o in y])
    ax2.set_xticks((0,1), (group0, group1))
    ax2.set_ylabel(f'{group1} Probability')
    ax2.set_ylim(0,1)
    ax2.set_title('B', loc = 'right')
    
    #coefficents
    lreg_coef = ['Logistic Regression'] + list(lreg.named_steps['logreg'].coef_[0])
    plsda_coef = ['PLS-DA'] + list(fit_plsda.named_steps['pls'].x_loadings_[:,0])
    index = [''] + predictors
    cols = [index, plsda_coef, lreg_coef]
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    for i, col in enumerate(cols):
        for j, elm in enumerate(col):
            if type(elm) != str:
                elm = "%.2f" % round(elm, 2)
            ax3.text(i/len(cols),
                     1-j/len(cols),
                     elm,
                     ha = 'center',
                     va = 'center')
    xlim = (-.15,0.9)
    ax3.plot(xlim,[0.9]*2,'-k', linewidth = 0.5)
    ax3.set_ylim(0,1.2)
    ax3.set_xlim(xlim)
    ax3.set_title('C', loc = 'right')
    return fig

data_dir = 'LipoCLEAN/build/'
qepro = read_data(data_dir + 'QE_Pro_model_training')
qepro = qepro[qepro['label'] == 1]
qepro['group'] = [0]*qepro.shape[0]
tof = read_data(data_dir + 'TOF_model_training')
tof = tof[tof['label'] == 1]
tof['group'] = [1]*tof.shape[0]
good = pd.concat((qepro,tof))

fig = plot_separability(good, 'Orbitrap', 'TOF')
for filetype in filetypes:
    fig.savefig(f'{fig_dir}FigureS4.{filetype}',
            bbox_inches = 'tight', dpi = DPI)

# =============================================================================
# Figure S5
# =============================================================================

data_dir = 'LipoCLEAN/build/'
good = read_data(data_dir + 'QE_Pro_model_training')
good = good[good['label'] == 1]
good['group'] = [int(f.startswith('build_data/QE')) for f in good['file']]

fig = plot_separability(good, 'OrbiPro', 'QE')
for filetype in filetypes:
    fig.savefig(f'{fig_dir}FigureS5.png',
            bbox_inches = 'tight', dpi = DPI)

# =============================================================================
# Figure S3
# =============================================================================

fig_dir = 'presentations_and_reports/paper/figures/'
fig1 = pyvips.Image.new_from_file(fig_dir + 'FigureS3.svg', dpi = DPI)
for filetype in filetypes:
    fig1.write_to_file(fig_dir + f'FigureS3.{filetype}')
