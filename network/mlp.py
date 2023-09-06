import torch
from torch import nn
import numpy as np

from .utils import combine_interleaved

class ImplicitNet(nn.Module):
    def __init__(
            self,
            d_in,
            dims,            
            skip_in=(),           
            d_out=1,
            geometric_init=True,
            bias=1.0,
            weight_norm=False,
            multires=0
    ):
        super().__init__()

        #dims = [d_in] + dims + [d_out + feature_vector_size]
        dims = [d_in] + dims + [d_out]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        #self.activation = nn.ReLU()
        self.activation = nn.Softplus(beta=100)
        #self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            conf.get_list("dims"),
            # skip_in=conf.get_list("skip_in",()),
            geometric_init = conf.get_bool("geometric_init",False),
            **kwargs
        )

class FCLayer(nn.Module):
    """
    Reference:
        https://github.com/vsitzmann/pytorch_prototyping/blob/10f49b1e7df38a58fd78451eac91d7ac1a21df64/pytorch_prototyping.py
    """
    def __init__(self, in_dim, out_dim, with_ln=True, use_softplus=False, non_linear=True):
        super().__init__()
        self.net = [nn.Linear(in_dim, out_dim)]
        if with_ln:
            self.net += [nn.LayerNorm([out_dim])]
        if non_linear:
            self.net += [torch.nn.LeakyReLU(negative_slope=0.2)] if not use_softplus else [nn.Softplus(beta=100)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x) 
    
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False,
                 with_ln=True,
                 use_softplus=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features, hidden_ch, with_ln, use_softplus))
        for i in range(num_hidden_layers):
            self.net.append(FCLayer(hidden_ch, hidden_ch, with_ln, use_softplus))
        if outermost_linear:
            self.net.append(Linear(hidden_ch, out_features))
        else:
            self.net.append(FCLayer(hidden_ch, out_features, with_ln, use_softplus))
        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self, item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)