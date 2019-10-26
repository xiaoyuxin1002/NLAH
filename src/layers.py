import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, alpha, device):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha
        self.device = device

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        
    def forward(self, features, nl_features, target_len, neighbor_len, adj, target_index_out, neighbor_index_target):
        
        h = torch.mm(features, self.W)
        a_input = torch.cat([h[adj[0]], h[adj[1]]],dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))
        
        nl_h = torch.mm(nl_features[neighbor_index_target], self.W)
        nl_a_input = torch.cat([h[neighbor_index_target], nl_h], dim=1)
        nl_e = self.leakyrelu(torch.matmul(nl_a_input, self.a).squeeze(1))
        
        attention = torch.full((target_len, neighbor_len+1), -9e15).to(self.device)
        attention[target_index_out, adj[1]] = e
        attention[:,-1] = nl_e
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)        

        h_prime = torch.matmul(attention[:,:-1], h) + attention[:,-1].reshape(-1,1)*nl_h 
        return F.elu(h_prime), F.elu(nl_h)
    
    
class MetapathAggreLayer(nn.Module):
      
    def __init__(self, nchannel, nhid, device, dropout):
        super(MetapathAggreLayer, self).__init__()
        
        self.nchannel = nchannel
        self.nhid = nhid
        self.device = device
        
        self.meta_att_vec = nn.Parameter(torch.zeros(size=(nchannel, nhid)))
        nn.init.xavier_uniform_(self.meta_att_vec.data, gain=1.414)
        
      
    def forward(self, hs, nnode): 
        
        meta_att = torch.empty((nnode, self.nchannel)).to(self.device)
        for i in range(self.nchannel):
            meta_att[:,i] = torch.mm(hs[i], self.meta_att_vec[i].view(-1,1)).squeeze(1)
        meta_att = F.softmax(meta_att, dim=1)
        
        aggre_hid = torch.empty((nnode, self.nhid)).to(self.device)        
        for i in range(nnode):
            aggre_hid[i,:] = torch.mm(meta_att[i,:].view(1,-1), hs[:,i,:])
            
        return aggre_hid