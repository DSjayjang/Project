import numpy as np
import torch

import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def descriptor_selection_3(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.SlogP_VSA10

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def descriptor_selection_5(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.SlogP_VSA10
        self_feats[i, 3] = mol_graph.NumAromaticRings
        self_feats[i, 4] = mol_graph.MaxEStateIndex

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
        
def descriptor_selection_7(samples):
    self_feats = np.empty((len(samples), 7), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.SlogP_VSA10
        self_feats[i, 3] = mol_graph.NumAromaticRings
        self_feats[i, 4] = mol_graph.MaxEStateIndex

        self_feats[i, 5] = mol_graph.PEOE_VSA14
        self_feats[i, 6] = mol_graph.fr_Ar_NH

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def descriptor_selection_10(samples):
    self_feats = np.empty((len(samples), 10), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.SlogP_VSA10
        self_feats[i, 3] = mol_graph.NumAromaticRings
        self_feats[i, 4] = mol_graph.MaxEStateIndex

        self_feats[i, 5] = mol_graph.PEOE_VSA14
        self_feats[i, 6] = mol_graph.fr_Ar_NH
        self_feats[i, 7] = mol_graph.SMR_VSA3
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.SlogP_VSA5

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def descriptor_selection_20(samples):
    self_feats = np.empty((len(samples), 20), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.SlogP_VSA10
        self_feats[i, 3] = mol_graph.NumAromaticRings
        self_feats[i, 4] = mol_graph.MaxEStateIndex

        self_feats[i, 5] = mol_graph.PEOE_VSA14
        self_feats[i, 6] = mol_graph.fr_Ar_NH
        self_feats[i, 7] = mol_graph.SMR_VSA3
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.SlogP_VSA5

        self_feats[i, 10] = mol_graph.VSA_EState8
        self_feats[i, 11] = mol_graph.MaxAbsEStateIndex
        self_feats[i, 12] = mol_graph.PEOE_VSA2
        self_feats[i, 13] = mol_graph.fr_Nhpyrrole
        self_feats[i, 14] = mol_graph.fr_amide

        self_feats[i, 15] = mol_graph.SlogP_VSA3
        self_feats[i, 16] = mol_graph.BCUT2D_MRHI
        self_feats[i, 17] = mol_graph.fr_nitrile
        self_feats[i, 18] = mol_graph.MolLogP
        self_feats[i, 19] = mol_graph.PEOE_VSA10

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)