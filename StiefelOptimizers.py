# --------------------------------------------------------
# This is an implementation for Variational Stiefel SGD/Adam in the paper
# Momentum Stiefel Optimizer, with Applications to
# Suitably-Orthogonal Attention, and Optimal Transport (ICLR 2023)
# https://arxiv.org/pdf/2205.14173.pdf
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from typing import List, Optional
from utils_StiefelOptimizers import *


inner_prod_param_dict={'Canonical':0.5, 'Euclidean':0.0}

class StiefelSGD(Optimizer):
    r""" Implementation of Momentum Stiefel (S)GD from the paper
    Momentum Stiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport (https://arxiv.org/abs/2205.14173)

    Purpose:
        Given a function f(X), find the minimum value of f under constraint that X has orthonormal columns
    Args:
        - params: A list of matrices. Containing parameters to optimize. 
        - lr: learning rate
        - momentum (float, optional): momentum factor (default: 0.9)
        - dampening (float, optional): dampening for momentum (default: 0)
        - expm_method (str in ['MatrixExp', 'Cayley', 'ForwardEuler'], optional): method to compute matrix exponential. (default: 'ForwardEuler')
        - inner_prod: ((float number less than 1 or string in `['Canonical', 'Euclidean']`, optional): the parameter in the canonical-type metric (defined in Definition 1 in the paper).
        - max_inner_iter: (int, optional): maximum number of iterations when computing matrix root inversion. (default: 100)
    Discussion: 
        - We recommend using the same hyperparameters when the model contains both Euclidean parameters and Stiefel parameters. See Remark 1 in the paper for details.
        - The matrices being optimized should have number of rows >= number of columns. Otherwise, the matrix will be transposed without warning. For tensors with more than 2 dimensions, all the dimensions will be flattened excepted the first dimension to create a matrix.
        - There is no significant difference when further tuning expm_method, inner_prod and max_inner_iter. Default is good enough to use.
        - No special orthonormal initialization for Stiefel matrices is required. Commonly used element-wise random Gaussian matrices will work and our optimizer will automatically project it onto the Stiefel manifold. However, explicit initialization using `torch.nn.init.orthogonal_` is still recommended.
    """
    def __init__(self, params, lr=required, momentum=0.9, dampening=0, expm_method='ForwardEuler', inner_prod='Canonical', max_inner_iter=100):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        assert expm_method in ['MatrixExp', 'Cayley', 'ForwardEuler'], 'expm_method not correct'

        # metric parameter in Definition 1 in the paper
        if isinstance(inner_prod, str):
            assert inner_prod in inner_prod_param_dict.keys(), 'inner_prod not correct'
            inner_prod_param=inner_prod_param_dict[inner_prod]
        else:
            inner_prod_param=float(inner_prod)
            assert inner_prod_param < 1
        self.a=inner_prod_param
        self.b=self.a/(self.a-1)

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, expm_method=expm_method, inner_prod=inner_prod, max_inner_iter=max_inner_iter)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            expm_method = group['expm_method']
            inner_prod=group['inner_prod']
            max_inner_iter=group['max_inner_iter']
            for X_raw in group['params']:
                if X_raw.grad is None:
                    continue
                # If X has more than 2 dimensions, all the dimensions except the first one will be flattened to make it a matrix.
                X=X_raw.view(X_raw.size()[0],-1)
                X_grad=X_raw.grad.view(X_raw.size()[0],-1)
                # X should be a tall and thin matrix (n>m). Otherwise, it will be transposed.
                if X.shape[0]<X.shape[1]:
                    X=X.transpose(0,1)
                    X_grad=X_grad.transpose(0,1)
                    transposed=True
                else:
                    transposed=False
                # Make the algorithm compatible with SO(n)
                # In that case, n=m, and we no longer need V
                n,m=X.shape
                if n==m:
                    square = True
                else:
                    square=False
                param_state = self.state[X_raw]
                # Same notation as in the paper are used for Y,V.
                if 'Y_buffer' not in param_state:
                    Y = param_state['Y_buffer']=torch.zeros(m,m, device=X.device, dtype=X.dtype)
                if 'V_buffer' not in param_state and not square:
                    V = param_state['V_buffer']=torch.zeros(n,m, device=X.device, dtype=X.dtype)
                if 't' not in param_state:
                    param_state['t']=0
                Y = param_state['Y_buffer']
                if not square:
                    V = param_state['V_buffer']
                Xt_Xgrad=torch.matmul(X.t(), X_grad)
                grad_Y=(1-self.b)/2*(Xt_Xgrad-Xt_Xgrad.t())
                if not square:
                    grad_V=X_grad-X@Xt_Xgrad
                # Dynamics phi_2 (will be skipped when n=m)
                if not square:
                    V.mul_(momentum)
                    V.add_(V@Y, alpha=-(3*self.a-2)/2*lr/momentum if momentum!=0 else 0.0)
                    V.add_(grad_V, alpha=-(1-dampening))
                # Dynamics phi_1
                Y.mul_(momentum).add_(grad_Y,alpha=-(1-dampening))
                if expm_method=='Cayley':
                    X.copy_(X.matmul(cayley(Y, alpha=lr)))
                elif expm_method=='MatrixExp':
                    X.copy_(X.matmul(torch.matrix_exp(lr*Y)))
                elif expm_method=='ForwardEuler':
                    X.add_(X@Y, alpha=lr)
                else:
                    raise NotImplementedError()
                # Dynamics phi_3 (will be skipped when n=m)
                if not square:
                    VTV=V.t()@V
                    XVTV=X@VTV
                    X.add_(V@(X.t()@X), alpha=lr)
                    V.add_(XVTV, alpha=-lr)
                X.copy_(X.matmul(matrix_root_inv(X.t()@X, iter_count=max_inner_iter)))
                # Check the structure for tangent bundle. For debug only. Please comment out.
                # assert torch.norm(X.t()@X-torch.eye(m, dtype=X.dtype, device=X.device))<torch.finfo(X.dtype).eps*torch.numel(Y)*10
                # assert torch.norm(Y.t()+Y)<torch.finfo(X.dtype).eps*torch.numel(Y)*10
                # assert torch.norm(X.t()@V)<torch.finfo(X.dtype).eps*torch.numel(Y)*10

                # reshape X back to its original shape
                if transposed:
                    X_raw.copy_(X.transpose(0,1).reshape(X_raw.shape))
                else:
                    X_raw.copy_(X.reshape(X_raw.shape))
                param_state['t']+=lr
        return loss


class StiefelAdam(Optimizer):
    r""" Implementation of Adam on Stiefel manifold from the paper
    Momentum Stiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport (https://arxiv.org/abs/2205.14173)
    Purpose:
        Given a function f(X), find the minimum value of f under constraint that X has orthonormal columns. This is the adaptive learning version. Suitable for machine learning problems.
    Args:
        - params: A list of matrices. Containing parameters to optimize. 
        - lr (float, optional): learning rate (default: 0.001)
        - betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        - expm_method (str in ['MatrixExp', 'Cayley', 'ForwardEuler'], optional): method to compute matrix exponential. (default: 'ForwardEuler')
        - inner_prod: (float number less than 1 or string in `['Canonical', 'Euclidean']`, optional): the parameter in the canonical-type metric (defined in Definition 1 in the paper).
        - max_inner_iter: (int, optional): maximum number of iterations when computing matrix root inversion. (default: 100)

    Discussion: 
        - We recommend using the same hyperparameters when the model contains both Euclidean parameters and Stiefel parameters. See Remark 1 in the paper for details.
        - The matrices being optimized should have number of rows >= number of columns. Otherwise, the matrix will be transposed without warning. For tensors with more than 2 dimensions, all the dimensions will be flattened excepted the first dimension to create a matrix.
        - There is no significant difference when further tuning expm_method, inner_prod and max_inner_iter. Default is good enough to use.
        - No special orthonormal initialization for Stiefel matrices is required. Commonly used element-wise random Gaussian matrices will work and our optimizer will automatically project it onto the Stiefel manifold. However, explicit initialization using `torch.nn.init.orthogonal_` is still recommended.
    """
    def __init__(self, params, lr=0.001, betas=(0.9,0.99), epsilon=1e-5, expm_method='ForwardEuler', inner_prod='Canonical', max_inner_iter=100):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        beta_1, beta_2=betas
        if beta_1<0 or beta_1>=1 or beta_2<=0 or beta_2>=1 :
            raise ValueError('beta out of range')
        assert expm_method in ['MatrixExp', 'Cayley', 'ForwardEuler'], 'expm_method not correct'
        if isinstance(inner_prod, str):
            assert inner_prod in inner_prod_param_dict.keys(), 'inner_prod not correct'
            inner_prod_param=inner_prod_param_dict[inner_prod]
        else:
            inner_prod_param=float(inner_prod)
            assert inner_prod_param < 1
        # metric parameter in Definition 1 in the paper
        self.a=inner_prod_param
        self.b=self.a/(self.a-1)

        defaults = dict(lr=lr, betas=betas, epsilon=epsilon, expm_method=expm_method, inner_prod=inner_prod, max_inner_iter=max_inner_iter)
        super(StiefelAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(StiefelAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for X_raw in group['params']:
                if X_raw.grad is None:
                    continue
                # If X has more than 2 dimensions, all the dimensions except the first one will be flattened to make it a matrix.
                X=X_raw.view(X_raw.size()[0],-1)
                X_grad=X_raw.grad.view(X_raw.size()[0],-1)
                # X should be a tall and thin matrix (n>m). Otherwise, it will be transposed.
                if X_raw.shape[0]<X_raw.shape[1]:
                    X=X_raw.transpose(0,1)
                    X_grad=X_grad.transpose(0,1)
                    transposed=True
                else:
                    transposed=False
                # Make the algorithm compatible with SO(n)
                # In that case, n=m, and we no longer need V
                beta_1, beta_2=group['betas']
                epsilon=group['epsilon']
                expm_method=group['expm_method']
                inner_prod=group['inner_prod']
                max_inner_iter=group['max_inner_iter']
                n,m=X.shape
                if n==m:
                    square = True
                else:
                    square=False
                param_state = self.state[X_raw]

                if 'Y_buffer' not in param_state:
                    Y = param_state['Y_buffer']=torch.zeros(m,m, device=X.device, dtype=X.dtype)
                if 'V_buffer' not in param_state and not square:
                    V = param_state['V_buffer']=torch.zeros(n,m, device=X.device, dtype=X.dtype)
                if 'p_Y_buffer' not in param_state:
                    p_Y = param_state['p_Y_buffer']=torch.zeros(m,m, device=X.device, dtype=X.dtype)
                if 'p_V_buffer' not in param_state and not square:
                    p_V = param_state['p_V_buffer']=torch.zeros(n,m, device=X.device, dtype=X.dtype)
                if 'step' not in param_state:
                    num_step=param_state['step']=0
                
                param_state['step']+=1
                Y = param_state['Y_buffer']
                p_Y = param_state['p_Y_buffer']
                if not square:
                    V = param_state['V_buffer']
                    p_V = param_state['p_V_buffer']
                step=param_state['step']

                bias_correction_1 = 1 - beta_1 ** step
                bias_correction_2 = 1 - beta_2 ** step

                Xt_Xgrad=torch.matmul(X.t(), X_grad)

                grad_Y=(1-self.b)/2*(Xt_Xgrad-Xt_Xgrad.t())
                if not square:
                    grad_V=-(X@Xt_Xgrad-X_grad)
                # Dynamics phi_2 (will be skipped when n=m)
                if not square:
                    p_V.mul_(beta_2).add_(grad_V**2, alpha=1-beta_2)
                # Dynamics phi_1
                Y.mul_(beta_1).add_(grad_Y, alpha=-(1-beta_1))
                p_Y.mul_(beta_2).add_(grad_Y**2, alpha=1-beta_2)
                denominator_Y=torch.sqrt(p_Y/bias_correction_2)+epsilon
                xi=lr/bias_correction_1*Y/denominator_Y
                if expm_method=='Cayley':
                    X.copy_(X.matmul(cayley(xi)))
                elif expm_method=='MatrixExp':
                    X.copy_(p.matmul(torch.matrix_exp(xi)))
                elif expm_method=='ForwardEuler':
                    X.add_(X@xi)
                else:
                    raise NotImplementedError()
                # Dynamics phi_3 (will be skipped when n=m)
                if not square:
                    V.mul_(beta_1)
                    V.add_(V@Y, alpha=-(3*self.a-2)/2*lr/beta_1 if beta_1!=0 else 0.0)
                    V.add_(grad_V, alpha=-(1-beta_1))
                    denominator_V=torch.sqrt(p_V/bias_correction_2)+epsilon
                    V_tilde=V/denominator_V-X@torch.linalg.inv(X.t()@X)@(X.t()@(V/denominator_V))
                    XVTV=X@(V_tilde.t()@V)
                    X.add_(V_tilde@(X.t()@X), alpha=lr)
                    V.add_(XVTV, alpha=-lr)

                X.copy_(X@matrix_root_inv(X.t()@X, iter_count=max_inner_iter))
                # Check the structure for tangent bundle. For debug only. Please comment out.
                # assert torch.norm(X.t()@X-torch.eye(m, dtype=X.dtype, device=X.device))<torch.finfo(X.dtype).eps*torch.numel(Y)*10
                # assert torch.norm(Y.t()+Y)<torch.finfo(X.dtype).eps*torch.numel(Y)*10
                # assert torch.norm(X.t()@V)<torch.finfo(X.dtype).eps*torch.numel(Y)*10

                # reshape X back to its original shape
                if transposed:
                    X_raw.copy_(X.transpose(0,1).reshape(X_raw.shape))
                else:
                    X_raw.copy_(X.reshape(X_raw.shape))
        
        return loss


class CombinedOptimizer(torch.optim.Optimizer):
    r"""
        This can be used when Euclidean and Stiefel parameters are contained in one model and are being optimized at the same time.
        This is due to that our StiefelSGD and Euclidean SGD (StiefelAdam and Euclidean Adam) uses the same hyperparameters and do not need to be tuned separately.
    """
    def __init__(self, *arg):
        self.optimizer_list=list(arg)
        param_group=[]
        for op in self.optimizer_list:
            for pg in op.param_groups:
                param_group.append(pg)
        super().__init__(param_group, defaults=dict())
    def zero_grad(self, set_to_none: bool = False):
        for op in self.optimizer_list:
            op.zero_grad()
    def step(self, closure):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for op in self.optimizer_list:
            loss=op.step()
        return loss
