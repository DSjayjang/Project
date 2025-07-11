import numpy as np
import torch

import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Freesolv
def collate_kfgcn_freesolv(samples):
    self_feats = np.empty((len(samples), 37), dtype=np.float32)

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
        # 21
        self_feats[i, 20] = mol_graph.PEOE_VSA7
        self_feats[i, 21] = mol_graph.MinPartialCharge
        self_feats[i, 22] = mol_graph.fr_aryl_methyl
        self_feats[i, 23] = mol_graph.NumSaturatedHeterocycles
        self_feats[i, 24] = mol_graph.NumHDonors
        # 26
        self_feats[i, 25] = mol_graph.fr_imidazole
        self_feats[i, 26] = mol_graph.fr_phos_ester
        self_feats[i, 27] = mol_graph.fr_Al_COO
        self_feats[i, 28] = mol_graph.EState_VSA6
        self_feats[i, 29] = mol_graph.PEOE_VSA8
        # 31
        self_feats[i, 30] = mol_graph.fr_ketone_Topliss
        self_feats[i, 31] = mol_graph.fr_imide
        self_feats[i, 32] = mol_graph.fr_nitro_arom_nonortho
        self_feats[i, 33] = mol_graph.EState_VSA8
        self_feats[i, 34] = mol_graph.fr_para_hydroxylation
        # 36
        self_feats[i, 35] = mol_graph.Kappa2
        self_feats[i, 36] = mol_graph.Ipc

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

# ESOL
def collate_kfgcn_esol(samples):
    self_feats = np.empty((len(samples), 43), dtype=np.float32)

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
        # 21
        self_feats[i, 20] = mol_graph.fr_C_O
        self_feats[i, 21] = mol_graph.fr_bicyclic
        self_feats[i, 22] = mol_graph.fr_ester
        self_feats[i, 23] = mol_graph.PEOE_VSA9
        self_feats[i, 24] = mol_graph.fr_Al_OH_noTert
        # 26
        self_feats[i, 25] = mol_graph.SlogP_VSA10
        self_feats[i, 26] = mol_graph.EState_VSA11
        self_feats[i, 27] = mol_graph.fr_imidazole
        self_feats[i, 28] = mol_graph.EState_VSA10
        self_feats[i, 29] = mol_graph.EState_VSA5
        # 31
        self_feats[i, 30] = mol_graph.SMR_VSA9
        self_feats[i, 31] = mol_graph.FractionCSP3
        self_feats[i, 32] = mol_graph.FpDensityMorgan2
        self_feats[i, 33] = mol_graph.fr_furan
        self_feats[i, 34] = mol_graph.fr_hdrzine
        # 36
        self_feats[i, 35] = mol_graph.fr_aryl_methyl
        self_feats[i, 36] = mol_graph.EState_VSA8
        self_feats[i, 37] = mol_graph.fr_phos_acid
        self_feats[i, 38] = mol_graph.SlogP_VSA7
        self_feats[i, 39] = mol_graph.SlogP_VSA4
        # 41
        self_feats[i, 40] = mol_graph.EState_VSA2
        self_feats[i, 41] = mol_graph.fr_nitro_arom_nonortho
        self_feats[i, 42] = mol_graph.fr_para_hydroxylation

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)