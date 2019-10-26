import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *


class JumpingKnowledgeNetwork(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, nlayer, nneighbor, dropout, alpha, device):
        super(JumpingKnowledgeNetwork, self).__init__()
        
        self.nneighbor = nneighbor
        self.nlayer = nlayer
        self.dropout = dropout
        
        self.att_layers = [GraphAttentionLayer(in_dim=nfeat, out_dim=nhid, dropout=dropout, alpha=alpha, device=device).to(device)]
        for _ in range(1, nlayer):
            self.att_layers.append(GraphAttentionLayer(in_dim=nhid, out_dim=nhid, dropout=dropout, alpha=alpha, device=device).to(device))
        
        for i, att in enumerate(self.att_layers):
            self.add_module('att_layer{}'.format(i), att)
            
    
    def sample(self, adj, samples):
        
        sample_list, adj_list = [samples], []
        for _ in range(self.nlayer):
            
            new_samples, new_adjs = set(sample_list[-1]), []
            for sample in sample_list[-1]:
                neighbor_size = adj[1][sample]
                start = adj[1][:sample].sum()
                
                if neighbor_size<=self.nneighbor:
                    curr_new_samples = adj[0][start:start+neighbor_size]                    
                else:
                    curr_new_samples = np.random.choice(adj[0][start:start+neighbor_size].tolist(), self.nneighbor, 
                                                        replace=False, p=adj[2][start:start+neighbor_size].tolist())
                new_samples = new_samples.union(set(curr_new_samples))
                curr_new_adjs = np.stack(([sample]*len(curr_new_samples), curr_new_samples), axis=-1).tolist()
                curr_new_adjs.append([sample, sample])
                new_adjs.append(curr_new_adjs)

            sample_list.append(np.array(list(new_samples)))
            adj_list.append(np.array([pair for chunk in new_adjs for pair in chunk]).T)
        
        return sample_list, adj_list
    
    
    def transform(self, sample_list, adj_list):
        
        trans_adj_list, target_index_outs, target_index_samples, neighbor_index_targets = [], [], [], []
        
        base_index_dict = {k:v for v,k in enumerate(sample_list[0])}        
        for i, adjs in enumerate(adj_list):
            
            target_index_outs.append([base_index_dict[k] for k in adjs[0]])
            target_index_samples.append([base_index_dict[k] for k in sample_list[0]])
            
            base_index_dict = {k:v for v,k in enumerate(sample_list[i+1])}
            
            neighbor_index_out, neighbor_index_in = [base_index_dict[k] for k in adjs[0]], [base_index_dict[k] for k in adjs[1]]
            trans_adj_list.append([neighbor_index_out, neighbor_index_in])            
            neighbor_index_targets.append([base_index_dict[k] for k in sample_list[i]])           
            
        return trans_adj_list, target_index_outs, target_index_samples, neighbor_index_targets            

      
    def forward(self, x, nl_x, adj, samples):
        
        sample_list, adj_list = self.sample(adj, samples)
        trans_adj_list, target_index_outs, target_index_samples, neighbor_index_targets = self.transform(sample_list, adj_list)

        outputs = []
        x, nl_x = x[sample_list[-1]], nl_x[sample_list[-1]]

        for i, att in enumerate(self.att_layers):

            x = F.dropout(x, self.dropout, training=self.training)
            nl_x = F.dropout(nl_x, self.dropout, training=self.training)
            
            x, nl_x = att(x, nl_x, len(sample_list[-i-2]), len(sample_list[-i-1]), 
                          trans_adj_list[-i-1], target_index_outs[-i-1], neighbor_index_targets[-i-1])
            
            outputs.append(x[target_index_samples[-i-1]])

        h = torch.stack(outputs, dim=0)
        h = torch.max(h, dim=0)[0]
        
        return h

    
class HeteroAttNet(nn.Module):
    
    def __init__(self, nchannel, nfeat, nhid, nclass, nlayer, nneighbor, dropout, alpha, device):
        super(HeteroAttNet, self).__init__()
        
        self.jknets = [JumpingKnowledgeNetwork(nfeat, nhid, nclass, nlayer, nneighbor[i], dropout, alpha, device) for i in range(nchannel)]
        self.agg_layer = MetapathAggreLayer(nchannel, nhid, device, dropout).to(device)
        self.linear_layer = torch.nn.Linear(nhid, nclass).to(device)
                             
        for i, jknet in enumerate(self.jknets):
            self.add_module('jknet_{}'.format(i), jknet)
        self.add_module("aggre", self.agg_layer)
        self.add_module('linear', self.linear_layer)
        
    def forward(self, x, nl_x, adjs, samples):
        
        metapath_out = []
        for i, jknet in enumerate(self.jknets):

            meta_x = jknet(x, nl_x[i], adjs[i], samples)
            metapath_out.append(meta_x)
        
        metapath_out = torch.stack(metapath_out, dim=0)        
        aggre_hid = self.agg_layer(metapath_out, len(samples))        
        pred = self.linear_layer(aggre_hid)
        pred = F.log_softmax(pred, dim=1)
        
        return pred