import os
import math
import time
import glob
import pickle
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from metrics import *
from models import *


seed = 100
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device=="cuda":
    torch.cuda.manual_seed(seed)
    
    
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, type=str, default='acm', help='Targeting dataset')
    parser.add_argument('-nl_type', required=True, type=str, default='2ndprox', help='Nonlocal measure')    
    parser.add_argument('-nhid', required=True, type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('-nlayer', required=True, type=int, default=2, help='Number of node-wise attention layers')
    parser.add_argument('-dropout', type=float, default=0.4, help='Dropout used in training')
    parser.add_argument('-alpha', type=float, default=0.2, help='Alpha used in LeakyReLU')
    parser.add_argument('-lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('-weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('-patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('-epochs', type=int, default=1000, help='Maximum training epochs')
    parser.add_argument('-min_sample', type=int, default=1, help='Minimum number of sampled neighbors in one metapath-based subgraph')
    parser.add_argument('-portion', default=1.0, help='Percentage of training or validation set used in training')
    
    return parser.parse_args() 


def load_dataset(dataset, nl_type):
    
    print('Loading dataset!')
    data = pickle.load(open("data/{}/all_data_{}.pkl".format(dataset, dataset), 'rb'))
    adjs = data['adjs']
    idx_train, idx_val, idx_test = data['idx_train'], data['idx_val'], data['idx_test']
    features = torch.from_numpy(data['features']).to(device)
    labels = torch.from_numpy(data['labels']).to(device)
    
    nl_features = pickle.load(open("data/{}/{}_data_{}.pkl".format(dataset, nl_type, dataset), 'rb'))
    nl_features = [torch.from_numpy(nl_feature).to(device) for nl_feature in nl_features]

    return adjs, features, nl_features, labels, idx_train, idx_val, idx_test


def train(model, optimizer, adjs, features, nl_features, labels, idx_train, idx_val, nsample, epoch):
    
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    real_idx_train = np.sort(idx_train[:nsample[0]])
    np.random.shuffle(idx_train)
    real_idx_val = np.sort(idx_val[:nsample[1]])
    np.random.shuffle(idx_val)

    samples = np.sort(np.concatenate([real_idx_train, real_idx_val]))
    index_dict = {k:v for v,k in enumerate(samples)}
    real_train_idx = [index_dict[k] for k in real_idx_train], 
    real_val_idx = [index_dict[k] for k in real_idx_val]
    
    output = model(features, nl_features, adjs, samples)
    
    loss_train = F.nll_loss(output[real_train_idx], labels[real_idx_train])
    loss_val = F.nll_loss(output[real_val_idx], labels[real_idx_val])
    nc_train = performance_classification(output[real_train_idx].detach(), labels[real_idx_train])
    nc_val = performance_classification(output[real_val_idx].detach(), labels[real_idx_val])
    
    loss_train.backward()
    optimizer.step()

    print('Epoch: {} ||'.format(epoch+1),
      'loss_train: {:.4f} acc_train: {:.4f} f1_train: {:.4f} ||'.format(loss_train.item(), nc_train[0], nc_train[1]),
      'loss_val: {:.4f} acc_val: {:.4f} f1_val: {:.4f} ||'.format(loss_val.item(), nc_val[0], nc_val[1]),
      'time: {:.4f}s'.format(time.time() - t))
    
    return loss_val.item()


def test(model, adjs, features, nl_features, labels, idx_test, nsample):
    
    real_idx_test = np.sort(idx_test[:nsample[2]])
    np.random.shuffle(idx_test)
    
    model.eval()
    output = model(features, nl_features, adjs, real_idx_test)
    
    loss_test = F.nll_loss(output, labels[real_idx_test])
    nc_test = performance_classification(output.detach(), labels[real_idx_test])
         
    print("Test set results: loss: {:.4f}, acc_test: {:.4f}, f1_test: {:.4f}".format(loss_test.item(), nc_test[0], nc_test[1]))
    
    return

    
def main():
    
    args = parse_args()
    adjs, features, nl_features, labels, idx_train, idx_val, idx_test = load_dataset(args.dataset, args.nl_type)

    nclass, nnode, nfeat, nchannel = int(labels.max())+1, features.shape[0], features.shape[1], len(adjs)    
    nneighbor = [max(int(np.log(len(adj[0])*2/nnode)), args.min_sample) for adj in adjs]   
    nsample = [int(len(idx_train)*args.portion), int(len(idx_val)*args.portion), len(idx_test)]
    
    t_total = time.time()    
    model = HeteroAttNet(nchannel, nfeat, args.nhid, nclass, args.nlayer, nneighbor, args.dropout, args.alpha, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    loss_values = []
    bad_counter, best, best_epoch = 0, args.epochs+1, 0
    for epoch in range(args.epochs):
    
        loss_values.append(train(model, optimizer, adjs, features, nl_features, labels, idx_train, idx_val, nsample, epoch))
        
        if loss_values[-1] < best:
            best, best_epoch, bad_counter = loss_values[-1], epoch, 0
        
            files = glob.glob('model/{}/{}/*.pkl'.format(args.dataset, args.nl_type))
            for file in files:
                os.remove(file)
            torch.save(model.state_dict(), 'model/{}/{}/{}.pkl'.format(args.dataset, args.nl_type, epoch))
        else:
            bad_counter += 1
            
        if bad_counter == args.patience: break
        
    print("NL Type {} Optimization Finished!".format(args.nl_type))
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('model/{}/{}/{}.pkl'.format(args.dataset, args.nl_type, best_epoch)))
    
    test(model, adjs, features, nl_features, labels, idx_test, nsample)   
    
    return
    
    
if __name__=='__main__':
    main()