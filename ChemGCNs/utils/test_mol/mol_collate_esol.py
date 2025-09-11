import numpy as np
import torch

import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_kfgcn_3(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 2] = mol_graph.MaxEStateIndex

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_5(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.SMR_VSA10
        self_feats[i, 4] = mol_graph.Kappa2

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
        
def collate_kfgcn_7(samples):
    self_feats = np.empty((len(samples), 7), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.SMR_VSA10
        self_feats[i, 4] = mol_graph.Kappa2

        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.PEOE_VSA13

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_10(samples):
    self_feats = np.empty((len(samples), 10), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.SMR_VSA10
        self_feats[i, 4] = mol_graph.Kappa2

        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.PEOE_VSA13
        self_feats[i, 7] = mol_graph.MinAbsPartialCharge
        self_feats[i, 8] = mol_graph.BCUT2D_CHGHI
        self_feats[i, 9] = mol_graph.PEOE_VSA6

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_20(samples):
    self_feats = np.empty((len(samples), 20), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.SMR_VSA10
        self_feats[i, 4] = mol_graph.Kappa2

        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.PEOE_VSA13
        self_feats[i, 7] = mol_graph.MinAbsPartialCharge
        self_feats[i, 8] = mol_graph.BCUT2D_CHGHI
        self_feats[i, 9] = mol_graph.PEOE_VSA6

        self_feats[i, 10] = mol_graph.SlogP_VSA1
        self_feats[i, 11] = mol_graph.fr_nitro
        self_feats[i, 12] = mol_graph.BalabanJ
        self_feats[i, 13] = mol_graph.SMR_VSA9
        self_feats[i, 14] = mol_graph.fr_alkyl_halide

        self_feats[i, 15] = mol_graph.fr_hdrzine
        self_feats[i, 16] = mol_graph.PEOE_VSA8
        self_feats[i, 17] = mol_graph.fr_Ar_NH
        self_feats[i, 18] = mol_graph.fr_imidazole
        self_feats[i, 19] = mol_graph.fr_Nhpyrrole

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)