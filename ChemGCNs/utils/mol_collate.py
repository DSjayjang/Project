import numpy as np
import torch

import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Freesolv
def descriptor_selection_freesolv(samples):
    self_feats = np.empty((len(samples), 50), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.SlogP_VSA10
        self_feats[i, 3] = mol_graph.NumAromaticRings
        self_feats[i, 4] = mol_graph.MaxEStateIndex
        # 6
        self_feats[i, 5] = mol_graph.PEOE_VSA14
        self_feats[i, 6] = mol_graph.fr_Ar_NH
        self_feats[i, 7] = mol_graph.SMR_VSA3
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.SlogP_VSA5
        # 11
        self_feats[i, 10] = mol_graph.VSA_EState8
        self_feats[i, 11] = mol_graph.MaxAbsEStateIndex
        self_feats[i, 12] = mol_graph.PEOE_VSA2
        self_feats[i, 13] = mol_graph.fr_Nhpyrrole
        self_feats[i, 14] = mol_graph.fr_amide
        # 16
        self_feats[i, 15] = mol_graph.SlogP_VSA3
        self_feats[i, 16] = mol_graph.BCUT2D_MRHI
        self_feats[i, 17] = mol_graph.fr_nitrile
        self_feats[i, 18] = mol_graph.MolLogP
        self_feats[i, 19] = mol_graph.PEOE_VSA10
        # 21
        self_feats[i, 20] = mol_graph.MinPartialCharge
        self_feats[i, 21] = mol_graph.fr_Al_OH
        self_feats[i, 22] = mol_graph.fr_sulfone
        self_feats[i, 23] = mol_graph.fr_Al_COO
        self_feats[i, 24] = mol_graph.fr_nitro_arom_nonortho
        # 26
        self_feats[i, 25] = mol_graph.fr_imidazole
        self_feats[i, 26] = mol_graph.fr_ketone_Topliss
        self_feats[i, 27] = mol_graph.PEOE_VSA7
        self_feats[i, 28] = mol_graph.fr_alkyl_halide
        self_feats[i, 29] = mol_graph.NumSaturatedHeterocycles
        # 31
        self_feats[i, 30] = mol_graph.fr_methoxy
        self_feats[i, 31] = mol_graph.fr_phos_acid
        self_feats[i, 32] = mol_graph.fr_pyridine
        self_feats[i, 33] = mol_graph.MinAbsEStateIndex
        self_feats[i, 34] = mol_graph.fr_para_hydroxylation
        # 36
        self_feats[i, 35] = mol_graph.fr_phos_ester
        self_feats[i, 36] = mol_graph.NumAromaticHeterocycles
        self_feats[i, 37] = mol_graph.PEOE_VSA8
        self_feats[i, 38] = mol_graph.fr_Ndealkylation2
        self_feats[i, 39] = mol_graph.PEOE_VSA5
        # 41
        self_feats[i, 40] = mol_graph.fr_aryl_methyl
        self_feats[i, 41] = mol_graph.NumHDonors
        self_feats[i, 42] = mol_graph.fr_imide
        self_feats[i, 43] = mol_graph.fr_priamide
        self_feats[i, 44] = mol_graph.RingCount
        # 46
        self_feats[i, 45] = mol_graph.SlogP_VSA8
        self_feats[i, 46] = mol_graph.VSA_EState4
        self_feats[i, 47] = mol_graph.SMR_VSA5
        self_feats[i, 48] = mol_graph.FpDensityMorgan3
        self_feats[i, 49] = mol_graph.FractionCSP3

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

# ESOL
def descriptor_selection_esol(samples):
    self_feats = np.empty((len(samples), 63), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.SMR_VSA10
        self_feats[i, 4] = mol_graph.Kappa2
        # 6
        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.PEOE_VSA13
        self_feats[i, 7] = mol_graph.MinAbsPartialCharge
        self_feats[i, 8] = mol_graph.BCUT2D_CHGHI
        self_feats[i, 9] = mol_graph.PEOE_VSA6
        # 11
        self_feats[i, 10] = mol_graph.SlogP_VSA1
        self_feats[i, 11] = mol_graph.fr_nitro
        self_feats[i, 12] = mol_graph.BalabanJ
        self_feats[i, 13] = mol_graph.SMR_VSA9
        self_feats[i, 14] = mol_graph.fr_alkyl_halide
        # 16
        self_feats[i, 15] = mol_graph.fr_hdrzine
        self_feats[i, 16] = mol_graph.PEOE_VSA8
        self_feats[i, 17] = mol_graph.fr_Ar_NH
        self_feats[i, 18] = mol_graph.fr_imidazole
        self_feats[i, 19] = mol_graph.fr_Nhpyrrole
        # 21
        self_feats[i, 20] = mol_graph.EState_VSA5
        self_feats[i, 21] = mol_graph.PEOE_VSA4
        self_feats[i, 22] = mol_graph.fr_ester
        self_feats[i, 23] = mol_graph.PEOE_VSA2
        self_feats[i, 24] = mol_graph.NumAromaticCarbocycles
        # 26
        self_feats[i, 25] = mol_graph.BCUT2D_LOGPHI
        self_feats[i, 26] = mol_graph.EState_VSA11
        self_feats[i, 27] = mol_graph.fr_furan
        self_feats[i, 28] = mol_graph.EState_VSA2
        self_feats[i, 29] = mol_graph.fr_benzene
        # 31
        self_feats[i, 30] = mol_graph.fr_sulfide
        self_feats[i, 31] = mol_graph.fr_aryl_methyl
        self_feats[i, 32] = mol_graph.SlogP_VSA10
        self_feats[i, 33] = mol_graph.HeavyAtomMolWt
        self_feats[i, 34] = mol_graph.fr_nitro_arom_nonortho
        # 36
        self_feats[i, 35] = mol_graph.FpDensityMorgan2
        self_feats[i, 36] = mol_graph.EState_VSA8
        self_feats[i, 37] = mol_graph.fr_bicyclic
        self_feats[i, 38] = mol_graph.fr_aniline
        self_feats[i, 39] = mol_graph.fr_allylic_oxid
        # 41
        self_feats[i, 40] = mol_graph.fr_C_S
        self_feats[i, 41] = mol_graph.SlogP_VSA7
        self_feats[i, 42] = mol_graph.SlogP_VSA4
        self_feats[i, 43] = mol_graph.fr_para_hydroxylation
        self_feats[i, 44] = mol_graph.PEOE_VSA7
        # 46
        self_feats[i, 45] = mol_graph.fr_Al_OH_noTert
        self_feats[i, 46] = mol_graph.fr_pyridine
        self_feats[i, 47] = mol_graph.fr_phos_acid
        self_feats[i, 48] = mol_graph.fr_phos_ester
        self_feats[i, 49] = mol_graph.NumAromaticHeterocycles
        # 51
        self_feats[i, 50] = mol_graph.EState_VSA7
        self_feats[i, 51] = mol_graph.PEOE_VSA12
        self_feats[i, 52] = mol_graph.Ipc
        self_feats[i, 53] = mol_graph.FpDensityMorgan1
        self_feats[i, 54] = mol_graph.PEOE_VSA14
        # 56
        self_feats[i, 55] = mol_graph.fr_guanido
        self_feats[i, 56] = mol_graph.fr_benzodiazepine
        self_feats[i, 57] = mol_graph.fr_thiophene
        self_feats[i, 58] = mol_graph.fr_Ndealkylation1
        self_feats[i, 59] = mol_graph.fr_aldehyde
        # 61
        self_feats[i, 60] = mol_graph.fr_term_acetylene
        self_feats[i, 61] = mol_graph.SMR_VSA2
        self_feats[i, 62] = mol_graph.fr_lactone

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

# ESOL
def descriptor_selection_lipo(samples):
    self_feats = np.empty((len(samples), 25), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.fr_COO
        self_feats[i, 2] = mol_graph.Ipc
        self_feats[i, 3] = mol_graph.fr_sulfonamd
        self_feats[i, 4] = mol_graph.PEOE_VSA7
        # 6
        self_feats[i, 5] = mol_graph.PEOE_VSA13
        self_feats[i, 6] = mol_graph.SlogP_VSA10
        self_feats[i, 7] = mol_graph.fr_unbrch_alkane
        self_feats[i, 8] = mol_graph.SMR_VSA10
        self_feats[i, 9] = mol_graph.PEOE_VSA12
        # 11
        self_feats[i, 10] = mol_graph.fr_guanido
        self_feats[i, 11] = mol_graph.FpDensityMorgan1
        self_feats[i, 12] = mol_graph.NHOHCount
        self_feats[i, 13] = mol_graph.fr_sulfide
        self_feats[i, 14] = mol_graph.VSA_EState5
        # 16
        self_feats[i, 15] = mol_graph.fr_HOCCN
        self_feats[i, 16] = mol_graph.fr_piperdine
        self_feats[i, 17] = mol_graph.NumSaturatedCarbocycles
        self_feats[i, 18] = mol_graph.fr_amidine
        self_feats[i, 19] = mol_graph.NumHDonors
        # 21
        self_feats[i, 20] = mol_graph.NumAromaticRings
        self_feats[i, 21] = mol_graph.BalabanJ
        self_feats[i, 22] = mol_graph.NumAromaticHeterocycles
        self_feats[i, 23] = mol_graph.MinEStateIndex
        self_feats[i, 24] = mol_graph.fr_Ar_N

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

# Self-Curated Gas
def descriptor_selection_scgas(samples):
    self_feats = np.empty((len(samples), 23), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.MolMR
        self_feats[i, 1] = mol_graph.TPSA
        self_feats[i, 2] = mol_graph.fr_halogen
        self_feats[i, 3] = mol_graph.SlogP_VSA12
        self_feats[i, 4] = mol_graph.RingCount
        # 6
        self_feats[i, 5] = mol_graph.Kappa1
        self_feats[i, 6] = mol_graph.NumHAcceptors
        self_feats[i, 7] = mol_graph.NumHDonors
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.SMR_VSA5
        # 11
        self_feats[i, 10] = mol_graph.Chi1
        self_feats[i, 11] = mol_graph.Chi3n
        self_feats[i, 12] = mol_graph.BertzCT
        self_feats[i, 13] = mol_graph.VSA_EState8
        self_feats[i, 14] = mol_graph.NumAliphaticCarbocycles
        # 16
        self_feats[i, 15] = mol_graph.HallKierAlpha
        self_feats[i, 16] = mol_graph.VSA_EState6
        self_feats[i, 17] = mol_graph.NumAromaticRings
        self_feats[i, 18] = mol_graph.Chi4n
        self_feats[i, 19] = mol_graph.PEOE_VSA7
        # 21
        self_feats[i, 20] = mol_graph.SlogP_VSA5
        self_feats[i, 21] = mol_graph.VSA_EState7
        self_feats[i, 22] = mol_graph.NOCount

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
