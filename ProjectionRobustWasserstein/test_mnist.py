import sys
sys.path.append('../optimizers')
sys.path.append('..')

from PRW import ProjectionRobustWasserstein
from Optimization.riemann_adap import RiemmanAdaptive
from Optimization.all import ProjectedStiefelSGD_algo, ProjectedStiefelAdam_algo, StiefelSGD_algo, StiefelAdam_algo, MomentumlessStiefelSGD_algo

# method_list = ['original_RGAS', 'original_RAGAS', 'StiefelSGD_ours', 'StiefelAdam_ours', 'MomentumlessStiefelSGD', 'ProjectedStiefelSGD', 'ProjectedStiefelAdam']
method_list = ['original_RGAS', 'original_RAGAS', 'StiefelSGD_ours', 'MomentumlessStiefelSGD', 'ProjectedStiefelSGD']

param_dict_dict={}
for method in method_list:
    param_dict_dict[method]=None


param_dict_dict['original_RGAS']={'lr':0.0008}
param_dict_dict['original_RAGAS']={'lr':0.01, 'beta':0.8}
param_dict_dict['StiefelSGD_ours']={'lr':0.001, 'momentum':0.5}
# param_dict_dict['StiefelAdam_ours']={'lr':0.00001, 'betas':(0.5, 0.8)}
param_dict_dict['MomentumlessStiefelSGD']={'lr':0.0006}
param_dict_dict['ProjectedStiefelSGD']={'lr':0.0007, 'momentum':0.5, 'stiefel':True}
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


class MNIST_NN(nn.Module):
    def __init__(self, num_class=10):
        super(MNIST_NN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # print(x.size())
        x = F.relu(x)
        feat = self.dropout2(x)
        x = self.fc2(feat)
        output = F.log_softmax(x, dim=1)
        return feat, output


def get_feats(feat_path):
    '''
    Computes MNIST features from CNN
    Pre-computed CNN models can be found in models/cnn_mnist.pt
    Extracted features can be found in results/exp5_mnist_feats.pkl
    '''

    model = MNIST_NN()
    model.load_state_dict(torch.load('models/cnn_mnist.pt'))
    model.eval()

    dset = datasets.MNIST('../data_set', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    # Set up dataloader
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    up = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True).float()

    def get_pred(x):
        x = up(x)
        feat, out = model(x)
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        score = torch.exp(out)

        return feat, pred, score

    N = len(dset)
    print(N)
    # Get predictions
    inception_preds = torch.zeros(N, 1)
    true_labels = torch.zeros(N, 1)
    feats = torch.zeros(N, 128)

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            imgs, lbs = batch
            imgs = imgs.float()
            batch_size_i = imgs.size(0)
            feat, pred, score = get_pred(imgs)
            inception_preds[i * batch_size:i * batch_size + batch_size_i] = pred
            true_labels[i * batch_size:i * batch_size + batch_size_i, 0] = lbs
            feats[i * batch_size:i * batch_size + batch_size_i, :] = feat

    ###############################
    ## Classification accuracy ####
    ###############################
    correct = inception_preds.eq(true_labels).sum().item()
    accuracy = correct / N
    print(accuracy)
    print(feats.size(), true_labels.size())
    # print(true_labels)

    feats_all = [[] for _ in range(10)]
    for j in range(feats.size(0)):
        lb = true_labels[j, 0].item()
        lb = int(lb)
        feats_all[lb].append(feats[j:j + 1, :])
    feats_all_t = []
    for feat in feats_all:
        feat = torch.cat(feat, 0)
        feat = feat.cpu().numpy()
        # print(feat.shape)
        feats_all_t.append(feat)

    with open(feat_path, 'wb') as f:
        pickle.dump(feats_all_t, f)


#########################################################################
# computer PRW distances between texts
#########################################################################
np.random.seed(357)

feat_path = './results/exp5_mnist_feats.pkl'

################################################
## Generate MNIST features of dim (128,)
################################################
get_feats(feat_path)

################################################
## Open MNIST features of dim (128,)
################################################

with open(feat_path, 'rb') as f:
    feats = pickle.load(f)

for feat in feats:
    print(feat.shape)

reg = 8
lr = 0.01
beta = 0.8

# method_list=['PRW', 'Ours']
OT_val_memory_dict_mnist={}
for method in method_list:
    OT_val_memory_dict_mnist[method]={}


d = 128  # dimension of MNIST features
k = 2

for method in method_list:
    t=time()
    for i in range(10):
        print((i,method))
        for j in range(i + 1, 10):
            assert i < j

            X = feats[i]
            Y = feats[j]

            na = X.shape[0]
            nb = Y.shape[0]

            a = (1. / na) * np.ones(na)
            b = (1. / nb) * np.ones(nb)
            # print(na,nb)
        
            algo=algo_dict[method](reg=reg, step_size_0=None, max_iter=50, threshold=0.001,
                            max_iter_sinkhorn=30,
                            threshold_sinkhorn=1e-3, use_gpu=False)
            PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
            param_dict=param_dict_dict[method]
            PRW.run(label_k_dict[method], param_dict)
            OT_val_memory_dict_mnist[method]['({}, {})'.format(i,j)]=PRW.get_maxmin_values()
# print latex scripts for table -- OT val

with open('OT_val_memory_dict_mnist.pkl', 'wb') as handle:
    pickle.dump(OT_val_memory_dict_mnist, handle)

