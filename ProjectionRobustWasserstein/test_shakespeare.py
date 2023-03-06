import sys
sys.path.append('../optimizers')
sys.path.append('..')

from PRW import ProjectionRobustWasserstein
from Optimization.riemann_adap import RiemmanAdaptive
from Optimization.all import ProjectedStiefelSGD_algo, ProjectedStiefelAdam_algo, StiefelSGD_algo, StiefelAdam_algo, MomentumlessStiefelSGD_algo
import io
# method_list = ['original_RGAS', 'original_RAGAS', 'StiefelSGD_ours', 'StiefelAdam_ours', 'MomentumlessStiefelSGD', 'ProjectedStiefelSGD', 'ProjectedStiefelAdam']
method_list = ['original_RGAS', 'original_RAGAS', 'StiefelSGD_ours', 'MomentumlessStiefelSGD', 'ProjectedStiefelSGD']
OT_val_memory_dict_shakespeare={}
for method in method_list:
    OT_val_memory_dict_shakespeare[method]={}
param_dict_dict={}
for method in method_list:
    param_dict_dict[method]=None


param_dict_dict['original_RGAS']={'lr':2.0}
param_dict_dict['original_RAGAS']={'lr':0.08, 'beta':0.9}
param_dict_dict['StiefelSGD_ours']={'lr':2.0, 'momentum':0.5}
# param_dict_dict['StiefelAdam_ours']={'lr':0.00001, 'betas':(0.5, 0.8)}
param_dict_dict['MomentumlessStiefelSGD']={'lr':2.0}
param_dict_dict['ProjectedStiefelSGD']={'lr':15.0, 'momentum':0.5, 'stiefel':True}
# param_dict_dict['ProjectedStiefelAdam']={'lr':0.05, 'momentum':0.5, 'beta2':0.8}

algo_dict={}
algo_dict['original_RGAS']=RiemmanAdaptive
algo_dict['original_RAGAS']=RiemmanAdaptive
algo_dict['StiefelSGD_ours']=StiefelSGD_algo
algo_dict['StiefelAdam_ours']=StiefelAdam_algo
algo_dict['MomentumlessStiefelSGD']=MomentumlessStiefelSGD_algo
algo_dict['ProjectedStiefelSGD']=ProjectedStiefelSGD_algo
algo_dict['ProjectedStiefelAdam']=ProjectedStiefelAdam_algo

label_k_dict={}
for method in method_list:
    label_k_dict[method]=None
label_k_dict['original_RGAS']=0
label_k_dict['original_RAGAS']=1



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################
# Experiment for showing PRW on learning topics on MNIST digits
############################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle
from time import time
from Optimization.frankwolfe import FrankWolfe

from PRW import ProjectionRobustWasserstein
from Optimization.riemann_adap import RiemmanAdaptive
import os
import pandas as pd

def load_vectors(fname, size=None):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 0
    for line in fin:
        if size and i >= size:
            break
        if i >= 2000:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(tokens[1:], dtype='f8')
        i += 1
    return data


import string
from collections import Counter

def textToMeasure(text):
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    words = text.split(' ')
    table = str.maketrans('', '', string.punctuation.replace("'", ""))
    words = [w.translate(table) for w in words if len(w) > 0]
    words = [w for w in words if w in dictionnary.keys()]
    words = [w for w in words if not w[0].isupper()]
    words = [w for w in words if not w.isdigit()]
    size = len(words)
    cX = Counter(words)
    words = list(set(words))
    a = np.array([cX[w] for w in words]) / size
    X = np.array([dictionnary[w] for w in words])
    return X, a, words


def load_text(file):
    """return X,a,words"""
    with open(file) as fp:
        text = fp.read()
    return textToMeasure(text)


if not os.path.isfile('Data/Text/wiki-news-300d-1M.vec'):
    print('[Warning]')
    print('Please download word vector at https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip')
    print('Put the unzipped "wiki-news-300d-1M.vec" in Data/Text folder, then try again')
    exit()

dictionnary = load_vectors('Data/Text/wiki-news-300d-1M.vec', size=20000)
dictionnary_pd = pd.DataFrame(dictionnary).T




#########################################################################
# Shakespeare plays are downloadable from
# https://www.folgerdigitaltexts.org/download/txt.html
# we pre-downloaded them in Data/Text/Shakespeare
#########################################################################
names = ['Henry V', 'Hamlet', 'Julius Caesar', 'The Merchant of Venice', 'Othello',
         'Romeo and Juliet']
scripts = ['H5.txt', 'Ham.txt', 'JC.txt', 'MV.txt', 'Oth.txt', 'Rom.txt']

assert len(names) == len(scripts)

Nb_scripts = len(scripts)
PRW_matrix = np.zeros((Nb_scripts, Nb_scripts))
measures = []
for art in scripts:
    measures.append(load_text('Data/Text/Shakespeare/' + art))

np.random.seed(357)

k = 2
reg = 0.2
for method in method_list:
    for art1 in scripts:
        for art2 in scripts:
            i = scripts.index(art1)
            j = scripts.index(art2)
            if i < j:
                X, a, words_X = measures[i]
                Y, b, words_Y = measures[j]

                algo=algo_dict[method](reg=reg, step_size_0=None, max_iter=50, threshold=0.001,
                                max_iter_sinkhorn=30,
                                threshold_sinkhorn=1e-3, use_gpu=False)
                PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
                param_dict=param_dict_dict[method]
                PRW.run(label_k_dict[method], param_dict)
                OT_val_memory_dict_shakespeare[method]['({}, {})'.format(i,j)]=PRW.get_maxmin_values()
                print(method, ' (', art1, ',', art2, ')')


with open('OT_val_memory_dict_shakespeare.pkl', 'wb') as handle:
    pickle.dump(OT_val_memory_dict_shakespeare, handle)

