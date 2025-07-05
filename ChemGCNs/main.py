import os
import pandas as pd
import numpy as np
from utils.config import SET_SEED, DATASET, BATCH_SIZE, MAX_EPOCHS, K
from utils.mol_dataset import MoleculeDataset

# 시드 고정
SET_SEED()

dataset = MoleculeDataset(DATASET)
dataset = dataset.samples








#################
import torch
import torch.nn as nn
import random
import dgl
from model import GCN
from model import EGCN
from model import trainer


# check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = BATCH_SIZE
max_epochs = MAX_EPOCHS
k = K

  

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    # batched_graph = dgl.batch(graphs)

    bg = dgl.batch(graphs).to(device)
    tgt = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
    return bg, tgt
    # return batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

########################################################################################################
# default
# don't touch
# ring 1개
def collate_emodel_ring(samples):
    self_feats = np.empty((len(samples), 1), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_rings

    graphs, labels = map(list, zip(*samples))

    bg = dgl.batch(graphs).to(device)
    # tgt = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
    # return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels).view(-1, 1).to(device)
    # return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
    return bg, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

# num_atoms + weight 2개
def collate_emodel_scale(samples):
    self_feats = np.empty((len(samples), 2), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight

    graphs, labels = map(list, zip(*samples))
    # batched_graph = dgl.batch(graphs)
    bg = dgl.batch(graphs).to(device)
    # tgt = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
    # return bg, tgt
    # return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels).view(-1, 1).to(device)
    # return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
    return bg, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)


# num_atoms + weight + ring 3개
def collate_emodel(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight
        self_feats[i, 2] = mol_graph.num_rings

    graphs, labels = map(list, zip(*samples))
    # batched_graph = dgl.batch(graphs)
    bg = dgl.batch(graphs).to(device)
    # tgt = torch.tensor(labels, dtype=torch.float32).to(device)
    # return bg, tgt
    # return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
    return bg, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

########################################################################################################


random.shuffle(dataset)

model_GCN = GCN.Net(8, 1).to(device)
model_EGCN_R = EGCN.Net(8, 1, 1).to(device)
model_EGCN_S = EGCN.Net(8, 1, 2).to(device)
model_EGCN = EGCN.Net(8, 1, 3).to(device)


# define loss function
criterion = nn.L1Loss(reduction='sum')
# criterion = nn.MSELoss(reduction='sum')

# train and evaluate competitors
test_losses = dict()


# #=====================================================================#
# # default model
# # don't touch

print('--------- GCN ---------')
test_losses['GCN'] = trainer.cross_validation(dataset, model_GCN, criterion, k, batch_size, max_epochs, trainer.train, trainer.test, collate)
print('test loss (GCN): ' + str(test_losses['GCN']))

print('--------- EGCN_RING ---------')
test_losses['EGCN_R'] = trainer.cross_validation(dataset, model_EGCN_R, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_ring)
print('test loss (EGCN_RING): ' + str(test_losses['EGCN_R']))

print('--------- EGCN_SCALE ---------')
test_losses['EGCN_S'] = trainer.cross_validation(dataset, model_EGCN_S, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_scale)
print('test loss (EGCN_SCALE): ' + str(test_losses['EGCN_S']))

print('--------- EGCN ---------')
test_losses['EGCN'] = trainer.cross_validation(dataset, model_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel)
print('test loss (EGCN): ' + str(test_losses['EGCN']))

print(test_losses)
