import dgl.function as fn
import torch 
import dgl 
import torch.nn.functional as F
import pickle 
import numpy as np
import time
import os
from dgl.nn.pytorch import Set2Set, NNConv, SetTransformerDecoder, AvgPooling, SumPooling, MaxPooling, SortPooling

class CG(torch.nn.Module):
    def __init__(self,
                atom_hidden_feats,
                bond_in_feats,
                 ):
        super(CG, self).__init__()
        self.atom_hidden_feats = atom_hidden_feats
        self.bond_in_feats = bond_in_feats
        self.lin = torch.nn.Linear(2*atom_hidden_feats+bond_in_feats,
                                   atom_hidden_feats,bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.LeakyReLU()

    def get_msg(self, edges):
        z = torch.cat([edges.src['v'], edges.dst['v'],edges.data['gdf_feat']], -1)
        z = self.lin(z)
        sig_z = self.sigmoid(z)
        softplus_z = self.softplus(z)
        return {'z':sig_z*softplus_z}
    def forward(self, graph, feat):
        graph = graph.local_var()
        graph.ndata['v'] = feat
        graph.update_all(message_func=self.get_msg,
                     reduce_func=fn.sum('z', 'm'))
        return graph.ndata['m']


class CGConvNet(torch.nn.Module):
    def __init__(self, atom_in_feats=100, 
                 atom_hidden_feats=64,
                 bond_in_feats=41,
                 n_conv=3,
                 n_h=1,
                 graph_rep_dim=64,
                 pooling='sum'
                ):
        super(CGConvNet, self).__init__()
        self.atom_in_feats = atom_in_feats
        self.atom_hidden_feats = atom_hidden_feats
        self.bond_in_feats = bond_in_feats
        self.n_conv = n_conv
        self.n_h = n_h
        self.graph_rep_dim = graph_rep_dim
        self.conv_layer = CG(atom_hidden_feats=atom_hidden_feats,
                             bond_in_feats = bond_in_feats
                            )
        if pooling=='sum':
            self.pooling_module = SumPooling()
        self.lin0 = torch.nn.Linear(atom_in_feats, atom_hidden_feats, bias=True)
        self.lin1 = torch.nn.Linear(atom_hidden_feats,graph_rep_dim,bias=True)
        self.lin2 = torch.nn.Linear(graph_rep_dim,1,bias=True)
        
    def forward(self,graph):
        n_feat = graph.ndata.pop('n_feat')   #B, N, atom_in_feats
        out = F.relu(self.lin0(n_feat))
        #v = out.unsqueeze(0) 
        for i in range(self.n_conv):
            #print(f"running iteration {i}")
            out = self.conv_layer(graph, out)
        graph_rep = F.relu(self.pooling_module(graph,out))
        preds = F.relu(self.lin2(graph_rep))
        return preds