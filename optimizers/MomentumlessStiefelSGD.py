# --------------------
# (S)GD on Stiefel manifold in `A feasible method for optimization with orthogonality constraints'
# (https://link.springer.com/article/10.1007/s10107-012-0584-1)
# This algorithm is termed as `Momentumless Stiefel SGD' in our paper
# --------------------

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from typing import List, Optional
# torch.set_default_tensor_type(torch.DoubleTensor)

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class MomentumlessStiefelSGD(Optimizer):
    def __init__(self, params, lr=required, method='NAG-SC', other_params=None, if_cayley=True):
        r'''
        Arguments:
        net: must be a plain fully connected nn. Recommand generated with class OrthogonalNN
        gamma: gamma in the AISTAT paper (momentum)

        '''
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, method=method, other_params=other_params, if_cayley=if_cayley)
        super(MomentumlessStiefelSGD, self).__init__(params, defaults)
    def __setstate__(self, state):
        super(MomentumlessStiefelSGD, self).__setstate__(state)

    @torch.no_grad()
    
    def step(self):
        """Performs a single optimization step.

        buf: xi in algorithm 2
        p: R in algotithm 2

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        for group in self.param_groups:
            lr = group['lr']
            
            for p_raw in group['params']:
                if p_raw.grad is None:
                    continue
                p=p_raw.view(-1, p_raw.shape[-2], p_raw.shape[-1])
                p_grad=p_raw.grad.view(-1, p_raw.shape[-2], p_raw.shape[-1])
                if p.shape[-2]<p.shape[-1]:
                    p=p.transpose(-2,-1)
                    p_grad=p_grad.transpose(-2,-1)
                    # print(torch.max(p_grad))                
                n,m=p.shape[-2], p.shape[-1]
                # print(p_grad)
                param_state = self.state[p_raw]
                

                
                if 2*m < n:
                    U = torch.cat((p_grad, p), dim=2)
                    V = torch.cat((p, -p_grad), dim=2)
                    p.add_( - lr * torch.bmm(torch.bmm(U, torch.linalg.inv(torch.eye(2*m, dtype=p.dtype, device=p.device)+lr/2*torch.bmm(V.transpose(-2,-1), U))), torch.bmm(V.transpose(-2,-1), p)))
                else:
                    W = torch.bmm(p, p_grad.T) - torch.bmm(p_grad , p.T)
                    p.copy_(torch.bmm(torch.bmm(torch.linalg.inv(torch.eye(n, dtype=p.dtype, device=p.device)+lr/2*W), (torch.eye(n, dtype=p.dtype, device=p.device)-lr/2*W), p)))
        return loss
