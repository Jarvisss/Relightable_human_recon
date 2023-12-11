import torch
import torch.nn as nn
# import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from .embedder import get_embedder

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

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

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class MLP(nn.Module):
    def __init__(self,
                 filter_channels,
                 name=None,
                 res_layers=[],
                 mean_layer=1,  ## if set to be -1, then no mean layer
                 norm='group',
                 activation='silu',
                 last_op=None,
                 output_bias=0.0,
                 output_scale=1.0,
                 num_views=1, 
                 use_mean_var=False,
                 use_feature_confidence=False,
                 output_mean_feat=False):

        super(MLP, self).__init__()
        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_views = num_views
        self.res_layers = res_layers #[1,2,3]
        self.mean_layer = mean_layer #[3]
        self.use_mean_var = use_mean_var
        self.use_feature_confidence = use_feature_confidence
        self.norm = norm
        self.last_op = last_op
        self.name = name
        self.output_mean_feat = output_mean_feat
        self.output_bias = output_bias
        self.output_scale = output_scale
        if activation == 'leaky':
            self.activate = nn.LeakyReLU()
        elif activation == 'relu':
            self.activate = nn.ReLU()
        elif activation == 'silu':
            self.activate = nn.SiLU()
        elif activation == 'softplus':
            self.activate = nn.Softplus(beta=100)

        for l in range(0, len(filter_channels) - 1):
            if l in self.res_layers:
                self.filters.append(
                    nn.Conv1d(filter_channels[l] + filter_channels[0],
                              filter_channels[l + 1], 1))
            else:
                self.filters.append(
                    nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))

            if l == self.mean_layer and self.use_feature_confidence:
                if l in self.res_layers:
                    self.feature_confidence_filter = nn.Conv1d(filter_channels[l] + filter_channels[0],1, 1) # [BK, C+C0, N] -> [BK, 1, N]
                else:
                    self.feature_confidence_filter = nn.Conv1d(filter_channels[l], 1, 1)  # [BK, C, N] -> [BK, 1, N]


            if l != len(filter_channels) - 2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[l + 1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(filter_channels[l + 1]))
                elif norm == 'instance':
                    self.norms.append(nn.InstanceNorm1d(filter_channels[l +
                                                                        1]))
                elif norm == 'weight':
                    self.filters[l] = nn.utils.weight_norm(self.filters[l],
                                                           name='weight')
                    # print(self.filters[l].weight_g.size(),
                    #       self.filters[l].weight_v.size())
        

    def forward(self, feature):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''

        if not self.use_mean_var:
            y = feature
            tmpy = feature
            geo_feat = feature
        else:
            mean_feat = feature.view(
                        -1, self.num_views, feature.shape[1], feature.shape[2]
                    ).mean(dim=1, keepdim=True).expand(-1, self.num_views, -1, -1).view(-1, feature.shape[1], feature.shape[2])
            if self.num_views>1:
                var_feat = feature.view(
                        -1, self.num_views, feature.shape[1], feature.shape[2]
                    ).var(dim=1, keepdim=True).expand(-1, self.num_views, -1, -1).view(-1, feature.shape[1], feature.shape[2])
            else:
                var_feat = torch.zeros_like(mean_feat)
            y = torch.cat((feature, mean_feat, var_feat), dim=1)
            tmpy = torch.cat((feature, mean_feat, var_feat), dim=1)
            geo_feat = torch.cat((feature, mean_feat, var_feat), dim=1)
        

        for i, f in enumerate(self.filters):
            
            if i == self.mean_layer and self.use_feature_confidence:
                fcy = self.feature_confidence_filter(y if i not in self.res_layers else torch.cat([y, tmpy], 1) )
            y = f(y if i not in self.res_layers else torch.cat([y, tmpy], 1))
            if i < len(self.filters) - 2:
                if self.norm not in ['batch', 'group', 'instance']:
                    y = self.activate(y)
                else:
                    y = self.activate(self.norms[i](y))
            
            if self.num_views > 1 and i == self.mean_layer:
                if not self.use_feature_confidence: # if not using confidence, use mean feature in the middle of MLP instead
                    y = y.view(
                        -1, self.num_views, y.shape[1], y.shape[2]
                    ).mean(dim=1)
                    tmpy = feature.view(
                        -1, self.num_views, feature.shape[1], feature.shape[2]
                    ).mean(dim=1)
                
                else:
                    assert y.shape[2]==fcy.shape[2]
                    y = y.view(-1, self.num_views, y.shape[1], y.shape[2]) # [B,K,C]
                    tmpy = feature.view(-1, self.num_views, feature.shape[1], feature.shape[2])
                    fcy = fcy.view(-1, self.num_views, fcy.shape[1], fcy.shape[2])
                    fcy = F.softmax(fcy, dim=1)
                    y = torch.sum(y * fcy, dim=1) # [B, C, N]
                    tmpy = torch.sum(tmpy * fcy, dim=1)

                    geo_feat = y
        
        y = self.output_scale * (y+self.output_bias)
        if self.last_op is not None:
            y = self.last_op(y)
        
        if self.output_mean_feat:

            return y, geo_feat
        else:
            return y