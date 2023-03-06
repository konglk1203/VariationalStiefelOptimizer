#from .optimizer import Optimizer, required
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

from gutils import unit
from gutils import gproj
from gutils import clip_by_norm
from gutils import xTy
from gutils import gexp
from gutils import gpt
from gutils import gpt2

import pdb

class SGDG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'. 

        If grassmann is True, the variables will be updated by SGD-G proposed 
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

        References:
           - Minhyung Cho and Jaehyung Lee, Riemannian approach to batch normalization
             (https://arxiv.org/abs/1709.09603)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case grassmann is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 grassmann=False, omega=0, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        grassmann=grassmann, omega=0, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            grassmann = group['grassmann']

            if grassmann:
                grad_clip = group['grad_clip']
                omega = group['omega']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    unity,_ = unit(p.data.view(p.size()[0],-1))
                    g = p.grad.data.view(p.size()[0],-1)

                    if omega != 0:
                      # L=|Y'Y-I|^2/2=|YY'-I|^2/2+c
                      # dL/dY=2(YY'Y-Y)
                      g.add_(2*omega, torch.mm(torch.mm(unity, unity.t()), unity) - unity)

                    h = gproj(unity, g)

                    if grad_clip is not None:
                        h_hat = clip_by_norm(h, grad_clip)
                    else:
                        h_hat = h

                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros(h_hat.size())
                        if p.is_cuda:
                          param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()

                    mom = param_state['momentum_buffer']
                    mom_new = momentum*mom - group['lr']*h_hat

                    p.data.copy_(gexp(unity, mom_new).view(p.size()))
                    mom.copy_(gpt(unity, mom_new))

            else:
                # This routine is from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
                weight_decay = group['weight_decay']
                dampening = group['dampening']
                nesterov = group['nesterov']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss

class AdamG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'. 

        If grassmann is True, the variables will be updated by Adam-G proposed 
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

        References:
           - Minhyung Cho and Jaehyung Lee, Riemannian approach to batch normalization
             (https://arxiv.org/abs/1709.09603)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use Adam-G (default: False)

        -- parameters in case grassmann is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        beta2 (float, optional): the exponential decay rate for the second moment estimates (defulat: 0.99)
        epsilon (float, optional): a small constant for numerical stability (default: 1e-8)
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, 
                 grassmann=False, beta2=0.99, epsilon=1e-8, omega=0, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, 
                        grassmann=grassmann, beta2=beta2, epsilon=epsilon, omega=0, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AdamG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            grassmann = group['grassmann']

            if grassmann:
                beta1 = group['momentum']
                beta2 = group['beta2']
                epsilon = group['epsilon']
                grad_clip = group['grad_clip']
                omega = group['omega']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    unity,_ = unit(p.data.view(p.size()[0],-1))
                    g = p.grad.data.view(p.size()[0],-1)

                    if omega != 0:
                      # L=|Y'Y-I|^2/2=|YY'-I|^2/2+c
                      # dL/dY=2(YY'Y-Y)
                      g.add_(2*omega, torch.mm(torch.mm(unity, unity.t()), unity) - unity)

                    h = gproj(unity, g)

                    if grad_clip is not None:
                        h_hat = clip_by_norm(h, grad_clip)
                    else:
                        h_hat = h

                    param_state = self.state[p]
                    if 'm_buffer' not in param_state:
                        size=p.size()
                        param_state['m_buffer'] = torch.zeros([size[0], int(np.prod(size[1:]))])
                        param_state['v_buffer'] = torch.zeros([size[0], 1])
                        if p.is_cuda:
                            param_state['m_buffer'] = param_state['m_buffer'].cuda()
                            param_state['v_buffer'] = param_state['v_buffer'].cuda()

                        param_state['beta1_power'] = beta1
                        param_state['beta2_power'] = beta2

                    m = param_state['m_buffer']
                    v = param_state['v_buffer']
                    beta1_power = param_state['beta1_power']
                    beta2_power = param_state['beta2_power']

                    mnew = beta1*m  + (1.0-beta1)*h_hat
                    vnew = beta2*v  + (1.0-beta2)*xTy(h_hat,h_hat)

                    alpha = np.sqrt(1.-beta2_power) / (1.-beta1_power)
                    deltas = mnew / vnew.add(epsilon).sqrt()
                    deltas.mul_(-alpha*group['lr'])

                    p.data.copy_(gexp(unity, deltas).view(p.size()))
                    m.copy_(gpt2(unity, mnew, deltas))
                    v.copy_(vnew)

                    param_state['beta1_power']*=beta1
                    param_state['beta2_power']*=beta2
            else:
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                dampening = group['dampening']
                nesterov = group['nesterov']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss       
