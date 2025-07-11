import numpy as np
import torch

import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GCN
def collate_gcn(samples):
    graphs, labels = map(list, zip(*samples))

    bg = dgl.batch(graphs).to(device)
    tgt = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
    return bg, tgt

# EGCN_R
def collate_egcn_ring(samples):
    self_feats = np.empty((len(samples), 1), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_rings

    graphs, labels = map(list, zip(*samples))

    bg = dgl.batch(graphs).to(device)
    tgt = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

    return bg, torch.tensor(self_feats).to(device), tgt

# EGCN_S
def collate_egcn_scale(samples):
    self_feats = np.empty((len(samples), 2), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight

    graphs, labels = map(list, zip(*samples))

    bg = dgl.batch(graphs).to(device)
    tgt = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

    return bg, torch.tensor(self_feats).to(device), tgt

# EGCN
def collate_egcn(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight
        self_feats[i, 2] = mol_graph.num_rings

    graphs, labels = map(list, zip(*samples))

    bg = dgl.batch(graphs).to(device)
    tgt = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

    return bg, torch.tensor(self_feats).to(device), tgt
