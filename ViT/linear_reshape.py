import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import math

class LinearReshaped(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_reshape=None, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weight_reshape==None:
            weight_reshape=(out_features, in_features)
        assert in_features*out_features==torch.prod(torch.Tensor(weight_reshape))
        self.weight = Parameter(torch.empty(weight_reshape, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Due to reshape, please use fan_out for init.
        bound = 1/math.sqrt(self.in_features)
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight.view((self.out_features, self.in_features)), self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'