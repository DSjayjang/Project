import numpy as np
import torch

import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_kfgcn_esol_3(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_esol_5(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 4] = mol_graph.BCUT2D_CHGHI

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
        
def collate_kfgcn_esol_7(samples):
    self_feats = np.empty((len(samples), 7), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 4] = mol_graph.BCUT2D_CHGHI
        # 6
        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.fr_imide

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_esol_10(samples):
    self_feats = np.empty((len(samples), 10), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 4] = mol_graph.BCUT2D_CHGHI
        # 6
        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.fr_imide
        self_feats[i, 7] = mol_graph.Kappa2
        self_feats[i, 8] = mol_graph.MinAbsPartialCharge
        self_feats[i, 9] = mol_graph.NumAromaticHeterocycles

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_kfgcn_esol_20(samples):
    self_feats = np.empty((len(samples), 20), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 4] = mol_graph.BCUT2D_CHGHI
        # 6
        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.fr_imide
        self_feats[i, 7] = mol_graph.Kappa2
        self_feats[i, 8] = mol_graph.MinAbsPartialCharge
        self_feats[i, 9] = mol_graph.NumAromaticHeterocycles
        # 11
        self_feats[i, 10] = mol_graph.SlogP_VSA1
        self_feats[i, 11] = mol_graph.fr_amide
        self_feats[i, 12] = mol_graph.BalabanJ
        self_feats[i, 13] = mol_graph.fr_Ar_NH
        self_feats[i, 14] = mol_graph.PEOE_VSA8
        # 16
        self_feats[i, 15] = mol_graph.NumSaturatedRings
        self_feats[i, 16] = mol_graph.fr_NH0
        self_feats[i, 17] = mol_graph.PEOE_VSA13
        self_feats[i, 18] = mol_graph.fr_barbitur
        self_feats[i, 19] = mol_graph.fr_alkyl_halide

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)