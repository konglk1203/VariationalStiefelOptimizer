# -*- coding: utf-8 -*-

import numpy as np
from .algorithm import Algorithm

from StiefelOptimizers import StiefelSGD, StiefelAdam
from MomentumlessStiefelSGD import MomentumlessStiefelSGD
from ProjectedStiefelOptimizer.stiefel_optimizer import SGDG, AdamG

import torch


class Algo_Optimizer(Algorithm):

    def __init__(self, reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu,
                 verbose=False, transpose=False):
        assert reg >= 0
        step_size_0 = None
        super().__init__(reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose)
        self.addon_vars = None
        self.optimizer=None
        self.default_dict=None
        self.transpose=transpose

    def initialize(self, a, b, X, Y, Omega, k):
        """Initialize Omega with the projection onto the subspace spanned by top-k eigenvectors of V_pi*, where pi* (=OT_plan) is the (classical) optimal transport plan."""
        d = X.shape[1]
        U_0 = np.eye(k)
        U = np.zeros((d, k))
        U[:k, :] = U_0

        self.U = U
        self.Omega = U.dot(U.T)
        d,k=U.shape
    # Riemann Grad with Regularized PRW, if reg>0
    def run_stiefel(self, a, b, X, Y, k, param_dict):
        if self.transpose:
            U_torch=torch.from_numpy(self.U).t()
        else:
            U_torch=torch.from_numpy(self.U)
        if param_dict==None:
            param_dict=self.default_dict
        optimizer=self.optimizer([U_torch], **param_dict)
        d,k=self.U.shape
        maxmin_values = []
        Omega = self.Omega

        OT_val_last=0

        for t in range(self.max_iter):
            # Optimal transport computation (Sinkhorn)
            C = self.Mahalanobis(X, Y, Omega)
            OT_val, OT_plan = self.OT(a, b, C)
            maxmin_values.append(OT_val)
            self.pi = OT_plan

            # Second-order moment of the displacements
            V = self.Vpi(X, Y, a, b, OT_plan)  # d x d

            # G_t
            U_torch_laststep=U_torch.detach().clone()
            if self.transpose:
                U_torch.grad=torch.from_numpy(-2 * V.dot(self.U)).t()
            else:
                U_torch.grad=torch.from_numpy(-2 * V.dot(self.U))
            optimizer.step()
            
            # stopping criteria
            gap = np.linalg.norm(U_torch_laststep-U_torch) / np.linalg.norm(U_torch_laststep)
            
            if gap < self.threshold:
            # if OT_val_last-OT_val>-1e-4*OT_val:
                # break
                pass
            OT_val_last=OT_val

            # update Omega
            Omega = self.U.dot(self.U.T)
            # print(np.linalg.norm(np.transpose(self.U)@self.U-np.eye(k)))
            assert np.linalg.norm(np.transpose(self.U)@self.U-np.eye(k))<1e-3
        # print('Method = ours   Num iter = '+str(t)+'  OT val = '+str(OT_val_last))
        return Omega, maxmin_values, t



class ProjectedStiefelSGD_algo(Algo_Optimizer):

    def __init__(self, reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu,
                 verbose=False):
        super().__init__(reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose, transpose=True)
        self.optimizer=SGDG
        self.default_dict={'lr':0.0005, 'momentum':0.5, 'stiefel':True}


class ProjectedStiefelAdam_algo(Algo_Optimizer):

    def __init__(self, reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu,
                 verbose=False):
        super().__init__(reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose, transpose=True)
        self.optimizer=AdamG
        self.default_dict={'lr':0.001}



class StiefelSGD_algo(Algo_Optimizer):

    def __init__(self, reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu,
                 verbose=False):
        super().__init__(reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose)
        self.optimizer=StiefelSGD
        self.default_dict={'lr':0.0005, 'momentum':0.5}

class StiefelAdam_algo(Algo_Optimizer):

    def __init__(self, reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu,
                 verbose=False):
        super().__init__(reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose)
        self.optimizer=StiefelAdam
        self.default_dict={'lr':0.0005, 'betas':(0.5, 0.5)}

class MomentumlessStiefelSGD_algo(Algo_Optimizer):

    def __init__(self, reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu,
                 verbose=False):
        super().__init__(reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose)
        self.optimizer=MomentumlessStiefelSGD
        self.default_dict={'lr':0.001}

    
