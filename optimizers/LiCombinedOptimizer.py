# -----------------
# a modified version of ProjectedStiefelOptimizer.py in [2]. Applying our retraction to optimizer in `Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform' (https://arxiv.org/abs/2002.01113)
# Only used in leading eigenvalue test. See Fig.6 in the paper for details.
# -----------------

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


def matrix_square_root(mat_a, mat_a_size, iter_count=100, ridge_epsilon=1e-4):
  """
  Stable iterations for the matrix square root, Nicholas J. Higham
  Page 231, Eq 2.6b
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.6.8799&rep=rep1&type=pdf
  """
  def _iter_condition(i, unused_mat_y, unused_old_mat_y, unused_mat_z,
                      unused_old_mat_z, err, old_err):
    # This method require that we check for divergence every step.
    return i < iter_count and err < old_err

  def _iter_body(i, mat_y, unused_old_mat_y, mat_z, unused_old_mat_z, err,
                 unused_old_err):
    current_iterate = 0.5 * (3.0 * identity - torch.matmul(mat_z, mat_y))
    current_mat_y = torch.matmul(mat_y, current_iterate)
    current_mat_z = torch.matmul(current_iterate, mat_z)
    # Compute the error in approximation.
    mat_sqrt_a = current_mat_y * torch.sqrt(norm)
    mat_a_approx = torch.matmul(mat_sqrt_a, mat_sqrt_a)
    residual = mat_a - mat_a_approx
    current_err = torch.norm(residual, p=2) / norm
    return i + 1, current_mat_y, mat_y, current_mat_z, mat_z, current_err, err

  identity = torch.eye(mat_a_size, device=mat_a.device, dtype=mat_a.dtype)
  mat_a = mat_a + ridge_epsilon * identity
  norm = torch.norm(mat_a, p=2)
  mat_init_y = mat_a / norm
  mat_init_z = identity
  init_err = norm

  func_input=[0, mat_init_y, mat_init_y, mat_init_z, mat_init_z, init_err, init_err + 1.0]
  while _iter_condition(*func_input):
    func_input=_iter_body(*func_input)
  return func_input[2] * torch.sqrt(norm), func_input[4] / torch.sqrt(norm)
def matrix_root(A):
  A_root, _ = matrix_square_root(A, A.shape[0], ridge_epsilon=0)
  return A_root

def matrix_root_inv(A, iter_count=100):
  _, A_root_inv = matrix_square_root(A, A.shape[0], ridge_epsilon=0, iter_count=iter_count)
  return A_root_inv


### Compute by SVD. Super expensive. For debug only

def mat_root_inv_for_debug(A):
    D, U=torch.symeig(A, eigenvectors=True)
    return U@torch.diag(1/torch.sqrt(D))@U.t()



def cayley(Y, alpha=1.0):
    return torch.linalg.inv(torch.eye(Y.shape[0],device=Y.device, dtype=Y.dtype).add(Y, alpha=-alpha/2))@(torch.eye(Y.shape[0],device=Y.device, dtype=Y.dtype).add(Y, alpha=alpha/2))




inner_prod_param_dict={'Canonical':0.5, 'Euclidean':0.0}



class LiCombinedOptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, dampening=0, expm_method='ForwardEuler', inner_prod='Canonical', max_inner_iter=100):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        assert expm_method in ['MatrixExp', 'Cayley', 'ForwardEuler'], 'expm_method not correct'
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
        """Performs a single optimization step.

        buf: xi in algorithm 2
        p: R in algotithm 2

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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
            for p_raw in group['params']:
                if p_raw.grad is None:
                    continue
                p=p_raw.view(p_raw.size()[0],-1)
                p_grad=p_raw.grad.view(p_raw.size()[0],-1)
                if p.shape[0]<p.shape[1]:
                    p=p.transpose(0,1)
                    p_grad=p_grad.transpose(0,1)
                    transposed=True
                else:
                    transposed=False
                    
                
                n,m=p.shape
                if n==m:
                    square = True
                else:
                    square=False
                param_state = self.state[p_raw]

                if 'M_buffer' not in param_state:
                    param_state['M_buffer']=torch.zeros(n,m, device=p.device, dtype=p.dtype)
                

                M = param_state['M_buffer']
                

                M.mul_(momentum).add_(p_grad,alpha=-(1-dampening))
                p.add_(M, alpha=lr)
                p.copy_(p.matmul(matrix_root_inv(p.t()@p, iter_count=max_inner_iter)))
                # assert torch.norm(p.t()@p-torch.eye(m, dtype=p.dtype, device=p.device))<torch.finfo(p.dtype).eps*torch.numel(Y)*10
                # assert torch.norm(Y.t()+Y)<torch.finfo(p.dtype).eps*torch.numel(Y)*10
                # assert torch.norm(p.t()@V)<torch.finfo(p.dtype).eps*torch.numel(Y)*10

                if transposed:
                    p_raw.copy_(p.transpose(0,1).reshape(p_raw.shape))
                else:
                    p_raw.copy_(p.reshape(p_raw.shape))


        return loss

