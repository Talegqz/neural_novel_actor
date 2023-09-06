import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_scatter import scatter


class GNN(nn.Module):
    def __init__(self,
                 args,
                 cfg,
                 device,
                 node_input_size = 19,
                 edge_input_size = 4,
                 latent_size = 128,
                 mlp_hidden_size = 128,
                 mlp_num_hidden_layers = 2,
                 num_message_passing_steps = 5,
                 output_size = 128
                 ):
        
        super().__init__()
        
        self.args = args
        self._node_input_size = node_input_size
        self._edge_input_size = edge_input_size
        self._latent_size = self.args.latent_codes_dim
        self._mlp_hidden_size = self.args.latent_codes_dim
        self._num_message_passing_steps = num_message_passing_steps
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._output_size = output_size
        self.args = args
        self.cfg = cfg
        self.device = device
        
        self._network_builder()

        
    def forward(self, input_graph):
        # Encode the input graph.
        latent_graph_0 = self._encode(input_graph)
        # Do `m` message passing steps in the latent graphs.
        latent_graph_m = self._process(latent_graph_0)
        return self._decode(latent_graph_m)

    
    def _network_builder(self):
        """Builds the networks."""

        global_dim = 0


        self.node_encoder = build_mlp_with_layer_norm(self._node_input_size + global_dim,num_hidden_layers=2,output_size=self.args.latent_codes_dim)


 
        self.edge_encoder = build_mlp_with_layer_norm(self._edge_input_size,num_hidden_layers=2,output_size=self.args.latent_codes_dim)

        self._processor_networks = nn.ModuleList()

        for _ in range(self._num_message_passing_steps):
            self._processor_networks.append(GCN(self.args.latent_codes_dim,self.args.latent_codes_dim,self.args.latent_codes_dim,num_layers=2,dropout=0.0,use_residual=False).to(self.device))

        self._decoder_network = build_mlp_with_layer_norm(self.args.latent_codes_dim,output_size=self.args.latent_codes_dim)
        
        return

            
    def _encode(self, input_graph):
        if input_graph['global'] is not None:
            input_graph['node_feat'] = torch.cat([input_graph['node_feat'], input_graph['global']], dim=-1)
            
        latent_graph_0 = {
            'node_feat': self.node_encoder(input_graph['node_feat']),
            'edge_feat': None,
            'global': None,
            'n_node': input_graph['n_node'],
            'n_edge': input_graph['n_edge'],
            'edge_index': input_graph['edge_index']
        }

        num_nodes = torch.sum(latent_graph_0['n_node']).item()
        latent_graph_0['edge_feat'] = self.edge_encoder(input_graph['edge_feat'])
        latent_graph_0['node_feat'] += scatter(latent_graph_0['edge_feat'], latent_graph_0['edge_index'][0],dim=0, dim_size=num_nodes, reduce='mean')

        return latent_graph_0


    
    def _decode(self, latent_graph):
        return self._decoder_network(latent_graph['node_feat'])
    
    def _process(self, latent_graph_0):
        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        for processor_network_k in self._processor_networks:
            latent_graph_k = self._process_step(
                processor_network_k, latent_graph_prev_k)
            latent_graph_prev_k = latent_graph_k

        latent_graph_m = latent_graph_k
        return latent_graph_m
        

    def _process_step(self, processor_network_k, latent_graph_prev_k):
        new_node_feature = processor_network_k(latent_graph_prev_k)
        latent_graph_k = {
            'node_feat': latent_graph_prev_k['node_feat'] + new_node_feature,
            'edge_feat': None,
            'global': None,
            'n_node': latent_graph_prev_k['n_node'],
            'n_edge': latent_graph_prev_k['n_edge'],
            'edge_index': latent_graph_prev_k['edge_index']
        }
        return latent_graph_k


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, use_residual=False):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = nn.LeakyReLU(0.05)
        self.use_bn = use_bn
        self.use_residual = use_residual

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data['node_feat']
        x = self.convs[0](x, data['edge_index'])
        x = self.activation(x)
        if self.use_bn:
            x = self.bns[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[1:-1]):
            x_ = conv(x, data['edge_index'])
            if self.use_bn:
                x_ = self.bns[i](x_)
            x_ = self.activation(x_)
            x_ = F.dropout(x_, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x_ + x
            else:
                x = x_
        x = self.convs[-1](x, data['edge_index'])
        return x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if num_hidden_layers == 0:
            self.lins.append(nn.Linear(input_size, output_size))
        else:
            self.lins.append(nn.Linear(input_size, hidden_size))
            self.lins.append(nn.LeakyReLU(0.05))
            for _ in range(num_hidden_layers - 1):
                self.lins.append(nn.Linear(hidden_size, hidden_size))
                self.lins.append(nn.LeakyReLU(0.05))
            self.lins.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for lin in self.lins:
            x = lin(x)
        return x

def build_mlp_with_layer_norm(input_size,hidden_size=128,output_size=128,num_hidden_layers=2):
    mlp = MLP(
        input_size=input_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        output_size=output_size
    )
    return nn.Sequential(*[mlp, nn.LayerNorm(output_size)])