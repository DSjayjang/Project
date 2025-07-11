import numpy as np
import torch

import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_kfgcn_freesolv_3(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SMR_VSA5
        self_feats[i, 2] = mol_graph.SlogP_VSA2

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_freesolv_5(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SMR_VSA5
        self_feats[i, 2] = mol_graph.SlogP_VSA2
        self_feats[i, 3] = mol_graph.TPSA
        self_feats[i, 4] = mol_graph.MaxEStateIndex

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_freesolv_7(samples):
    self_feats = np.empty((len(samples), 7), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SMR_VSA5
        self_feats[i, 2] = mol_graph.SlogP_VSA2
        self_feats[i, 3] = mol_graph.TPSA
        self_feats[i, 4] = mol_graph.MaxEStateIndex
        # 6
        self_feats[i, 5] = mol_graph.fr_Ar_NH
        self_feats[i, 6] = mol_graph.Chi2v

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_freesolv_10(samples):
    self_feats = np.empty((len(samples), 10), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SMR_VSA5
        self_feats[i, 2] = mol_graph.SlogP_VSA2
        self_feats[i, 3] = mol_graph.TPSA
        self_feats[i, 4] = mol_graph.MaxEStateIndex
        # 6
        self_feats[i, 5] = mol_graph.fr_Ar_NH
        self_feats[i, 6] = mol_graph.Chi2v
        self_feats[i, 7] = mol_graph.SlogP_VSA10
        self_feats[i, 8] = mol_graph.NumHeteroatoms
        self_feats[i, 9] = mol_graph.RingCount

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_freesolv_20(samples):
    self_feats = np.empty((len(samples), 20), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SMR_VSA5
        self_feats[i, 2] = mol_graph.SlogP_VSA2
        self_feats[i, 3] = mol_graph.TPSA
        self_feats[i, 4] = mol_graph.MaxEStateIndex
        # 6
        self_feats[i, 5] = mol_graph.fr_Ar_NH
        self_feats[i, 6] = mol_graph.Chi2v
        self_feats[i, 7] = mol_graph.SlogP_VSA10
        self_feats[i, 8] = mol_graph.NumHeteroatoms
        self_feats[i, 9] = mol_graph.RingCount
        # 11
        self_feats[i, 10] = mol_graph.fr_amide
        self_feats[i, 11] = mol_graph.NumAromaticHeterocycles
        self_feats[i, 12] = mol_graph.PEOE_VSA14
        self_feats[i, 13] = mol_graph.SlogP_VSA4
        self_feats[i, 14] = mol_graph.VSA_EState8
        # 16
        self_feats[i, 15] = mol_graph.PEOE_VSA2
        self_feats[i, 16] = mol_graph.PEOE_VSA10
        self_feats[i, 17] = mol_graph.fr_Al_OH
        self_feats[i, 18] = mol_graph.fr_bicyclic
        self_feats[i, 19] = mol_graph.SMR_VSA2

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)