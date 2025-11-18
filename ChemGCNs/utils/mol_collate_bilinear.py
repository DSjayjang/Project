import numpy as np
import torch

import dgl
import pandas as pd
import rdkit.Chem.Descriptors as dsc

from utils.utils import FeatureNormalization
from utils.mol_graph import smiles_to_mol_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Freesolv
def descriptor_selection_freesolv_bilinear(samples):
    self_feats = np.empty((len(samples), 208), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.MaxEStateIndex
        self_feats[i, 1] = mol_graph.MinEStateIndex
        self_feats[i, 2] = mol_graph.MaxAbsEStateIndex
        self_feats[i, 3] = mol_graph.MinAbsEStateIndex
        self_feats[i, 4] = mol_graph.qed
        # 6
        self_feats[i, 5] = mol_graph.MolWt
        self_feats[i, 6] = mol_graph.HeavyAtomMolWt
        self_feats[i, 7] = mol_graph.ExactMolWt
        self_feats[i, 8] = mol_graph.NumValenceElectrons
        self_feats[i, 9] = mol_graph.NumRadicalElectrons
        # 11
        self_feats[i, 10] = mol_graph.MaxPartialCharge
        self_feats[i, 11] = mol_graph.MinPartialCharge
        self_feats[i, 12] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 13] = mol_graph.MinAbsPartialCharge
        self_feats[i, 14] = mol_graph.FpDensityMorgan1
        # 16
        self_feats[i, 15] = mol_graph.FpDensityMorgan2
        self_feats[i, 16] = mol_graph.FpDensityMorgan3
        self_feats[i, 17] = mol_graph.BCUT2D_MWHI
        self_feats[i, 18] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 19] = mol_graph.BCUT2D_CHGHI
        # 21
        self_feats[i, 20] = mol_graph.BCUT2D_CHGLO
        self_feats[i, 21] = mol_graph.BCUT2D_LOGPHI
        self_feats[i, 22] = mol_graph.BCUT2D_LOGPLOW
        self_feats[i, 23] = mol_graph.BCUT2D_MRHI
        self_feats[i, 24] = mol_graph.BCUT2D_MRLOW
        # 26
        self_feats[i, 25] = mol_graph.BalabanJ
        self_feats[i, 26] = mol_graph.BertzCT
        self_feats[i, 27] = mol_graph.Chi0
        self_feats[i, 28] = mol_graph.Chi0n
        self_feats[i, 29] = mol_graph.Chi0v
        # 31
        self_feats[i, 30] = mol_graph.Chi1
        self_feats[i, 31] = mol_graph.Chi1n
        self_feats[i, 32] = mol_graph.Chi1v
        self_feats[i, 33] = mol_graph.Chi2n
        self_feats[i, 34] = mol_graph.Chi2v
        # 36
        self_feats[i, 35] = mol_graph.Chi3n
        self_feats[i, 36] = mol_graph.Chi3v
        self_feats[i, 37] = mol_graph.Chi4n
        self_feats[i, 38] = mol_graph.Chi4v
        self_feats[i, 39] = mol_graph.HallKierAlpha
        # 41
        self_feats[i, 40] = mol_graph.Ipc
        self_feats[i, 41] = mol_graph.Kappa1
        self_feats[i, 42] = mol_graph.Kappa2
        self_feats[i, 43] = mol_graph.Kappa3
        self_feats[i, 44] = mol_graph.LabuteASA
        # 46
        self_feats[i, 45] = mol_graph.PEOE_VSA1
        self_feats[i, 46] = mol_graph.PEOE_VSA10
        self_feats[i, 47] = mol_graph.PEOE_VSA11
        self_feats[i, 48] = mol_graph.PEOE_VSA12
        self_feats[i, 49] = mol_graph.PEOE_VSA13
        # 51
        self_feats[i, 50] = mol_graph.PEOE_VSA14
        self_feats[i, 51] = mol_graph.PEOE_VSA2
        self_feats[i, 52] = mol_graph.PEOE_VSA3
        self_feats[i, 53] = mol_graph.PEOE_VSA4
        self_feats[i, 54] = mol_graph.PEOE_VSA5
        # 56
        self_feats[i, 55] = mol_graph.PEOE_VSA6
        self_feats[i, 56] = mol_graph.PEOE_VSA7
        self_feats[i, 57] = mol_graph.PEOE_VSA8
        self_feats[i, 58] = mol_graph.PEOE_VSA9
        self_feats[i, 59] = mol_graph.SMR_VSA1
        # 61
        self_feats[i, 60] = mol_graph.SMR_VSA10
        self_feats[i, 61] = mol_graph.SMR_VSA2
        self_feats[i, 62] = mol_graph.SMR_VSA3
        self_feats[i, 63] = mol_graph.SMR_VSA4
        self_feats[i, 64] = mol_graph.SMR_VSA5
        # 66
        self_feats[i, 65] = mol_graph.SMR_VSA6
        self_feats[i, 66] = mol_graph.SMR_VSA7
        self_feats[i, 67] = mol_graph.SMR_VSA8
        self_feats[i, 68] = mol_graph.SMR_VSA9
        self_feats[i, 69] = mol_graph.SlogP_VSA1
        # 71
        self_feats[i, 70] = mol_graph.SlogP_VSA10
        self_feats[i, 71] = mol_graph.SlogP_VSA11
        self_feats[i, 72] = mol_graph.SlogP_VSA12
        self_feats[i, 73] = mol_graph.SlogP_VSA2
        self_feats[i, 74] = mol_graph.SlogP_VSA3
        # 76
        self_feats[i, 75] = mol_graph.SlogP_VSA4
        self_feats[i, 76] = mol_graph.SlogP_VSA5
        self_feats[i, 77] = mol_graph.SlogP_VSA6
        self_feats[i, 78] = mol_graph.SlogP_VSA7
        self_feats[i, 79] = mol_graph.SlogP_VSA8
        # 81
        self_feats[i, 80] = mol_graph.SlogP_VSA9
        self_feats[i, 81] = mol_graph.TPSA
        self_feats[i, 82] = mol_graph.EState_VSA1
        self_feats[i, 83] = mol_graph.EState_VSA10
        self_feats[i, 84] = mol_graph.EState_VSA11
        # 86
        self_feats[i, 85] = mol_graph.EState_VSA2
        self_feats[i, 86] = mol_graph.EState_VSA3
        self_feats[i, 87] = mol_graph.EState_VSA4
        self_feats[i, 88] = mol_graph.EState_VSA5
        self_feats[i, 89] = mol_graph.EState_VSA6
        # 91
        self_feats[i, 90] = mol_graph.EState_VSA7
        self_feats[i, 91] = mol_graph.EState_VSA8
        self_feats[i, 92] = mol_graph.EState_VSA9
        self_feats[i, 93] = mol_graph.VSA_EState1
        self_feats[i, 94] = mol_graph.VSA_EState10
        # 96
        self_feats[i, 95] = mol_graph.VSA_EState2
        self_feats[i, 96] = mol_graph.VSA_EState3
        self_feats[i, 97] = mol_graph.VSA_EState4
        self_feats[i, 98] = mol_graph.VSA_EState5
        self_feats[i, 99] = mol_graph.VSA_EState6
        # 101
        self_feats[i, 100] = mol_graph.VSA_EState7
        self_feats[i, 101] = mol_graph.VSA_EState8
        self_feats[i, 102] = mol_graph.VSA_EState9
        self_feats[i, 103] = mol_graph.FractionCSP3
        self_feats[i, 104] = mol_graph.HeavyAtomCount
        # 106
        self_feats[i, 105] = mol_graph.NHOHCount
        self_feats[i, 106] = mol_graph.NOCount
        self_feats[i, 107] = mol_graph.NumAliphaticCarbocycles
        self_feats[i, 108] = mol_graph.NumAliphaticHeterocycles
        self_feats[i, 109] = mol_graph.NumAliphaticRings
        # 111
        self_feats[i, 110] = mol_graph.NumAromaticCarbocycles
        self_feats[i, 111] = mol_graph.NumAromaticHeterocycles
        self_feats[i, 112] = mol_graph.NumAromaticRings
        self_feats[i, 113] = mol_graph.NumHAcceptors
        self_feats[i, 114] = mol_graph.NumHDonors
        # 116
        self_feats[i, 115] = mol_graph.NumHeteroatoms
        self_feats[i, 116] = mol_graph.NumRotatableBonds
        self_feats[i, 117] = mol_graph.NumSaturatedCarbocycles
        self_feats[i, 118] = mol_graph.NumSaturatedHeterocycles
        self_feats[i, 119] = mol_graph.NumSaturatedRings
        # 121
        self_feats[i, 120] = mol_graph.RingCount
        self_feats[i, 121] = mol_graph.MolLogP
        self_feats[i, 122] = mol_graph.MolMR
        self_feats[i, 123] = mol_graph.fr_Al_COO
        self_feats[i, 124] = mol_graph.fr_Al_OH
        # 126
        self_feats[i, 125] = mol_graph.fr_Al_OH_noTert
        self_feats[i, 126] = mol_graph.fr_ArN
        self_feats[i, 127] = mol_graph.fr_Ar_COO
        self_feats[i, 128] = mol_graph.fr_Ar_N
        self_feats[i, 129] = mol_graph.fr_Ar_NH
        # 131
        self_feats[i, 130] = mol_graph.fr_Ar_OH
        self_feats[i, 131] = mol_graph.fr_COO
        self_feats[i, 132] = mol_graph.fr_COO2
        self_feats[i, 133] = mol_graph.fr_C_O
        self_feats[i, 134] = mol_graph.fr_C_O_noCOO
        # 136
        self_feats[i, 135] = mol_graph.fr_C_S
        self_feats[i, 136] = mol_graph.fr_HOCCN
        self_feats[i, 137] = mol_graph.fr_Imine
        self_feats[i, 138] = mol_graph.fr_NH0
        self_feats[i, 139] = mol_graph.fr_NH1
        # 141
        self_feats[i, 140] = mol_graph.fr_NH2
        self_feats[i, 141] = mol_graph.fr_N_O
        self_feats[i, 142] = mol_graph.fr_Ndealkylation1
        self_feats[i, 143] = mol_graph.fr_Ndealkylation2
        self_feats[i, 144] = mol_graph.fr_Nhpyrrole
        # 146
        self_feats[i, 145] = mol_graph.fr_SH
        self_feats[i, 146] = mol_graph.fr_aldehyde
        self_feats[i, 147] = mol_graph.fr_alkyl_carbamate
        self_feats[i, 148] = mol_graph.fr_alkyl_halide
        self_feats[i, 149] = mol_graph.fr_allylic_oxid
        # 151
        self_feats[i, 150] = mol_graph.fr_amide
        self_feats[i, 151] = mol_graph.fr_amidine
        self_feats[i, 152] = mol_graph.fr_aniline
        self_feats[i, 153] = mol_graph.fr_aryl_methyl
        self_feats[i, 154] = mol_graph.fr_azide
        # 156
        self_feats[i, 155] = mol_graph.fr_azo
        self_feats[i, 156] = mol_graph.fr_barbitur
        self_feats[i, 157] = mol_graph.fr_benzene
        self_feats[i, 158] = mol_graph.fr_benzodiazepine
        self_feats[i, 159] = mol_graph.fr_bicyclic
        # 161
        self_feats[i, 160] = mol_graph.fr_diazo
        self_feats[i, 161] = mol_graph.fr_dihydropyridine
        self_feats[i, 162] = mol_graph.fr_epoxide
        self_feats[i, 163] = mol_graph.fr_ester
        self_feats[i, 164] = mol_graph.fr_ether
        # 166
        self_feats[i, 165] = mol_graph.fr_furan
        self_feats[i, 166] = mol_graph.fr_guanido
        self_feats[i, 167] = mol_graph.fr_halogen
        self_feats[i, 168] = mol_graph.fr_hdrzine
        self_feats[i, 169] = mol_graph.fr_hdrzone
        # 171
        self_feats[i, 170] = mol_graph.fr_imidazole
        self_feats[i, 171] = mol_graph.fr_imide
        self_feats[i, 172] = mol_graph.fr_isocyan
        self_feats[i, 173] = mol_graph.fr_isothiocyan
        self_feats[i, 174] = mol_graph.fr_ketone
        # 176
        self_feats[i, 175] = mol_graph.fr_ketone_Topliss
        self_feats[i, 176] = mol_graph.fr_lactam
        self_feats[i, 177] = mol_graph.fr_lactone
        self_feats[i, 178] = mol_graph.fr_methoxy
        self_feats[i, 179] = mol_graph.fr_morpholine
        # 181
        self_feats[i, 180] = mol_graph.fr_nitrile
        self_feats[i, 181] = mol_graph.fr_nitro
        self_feats[i, 182] = mol_graph.fr_nitro_arom
        self_feats[i, 183] = mol_graph.fr_nitro_arom_nonortho
        self_feats[i, 184] = mol_graph.fr_nitroso
        # 186
        self_feats[i, 185] = mol_graph.fr_oxazole
        self_feats[i, 186] = mol_graph.fr_oxime
        self_feats[i, 187] = mol_graph.fr_para_hydroxylation
        self_feats[i, 188] = mol_graph.fr_phenol
        self_feats[i, 189] = mol_graph.fr_phenol_noOrthoHbond
        # 191
        self_feats[i, 190] = mol_graph.fr_phos_acid
        self_feats[i, 191] = mol_graph.fr_phos_ester
        self_feats[i, 192] = mol_graph.fr_piperdine
        self_feats[i, 193] = mol_graph.fr_piperzine
        self_feats[i, 194] = mol_graph.fr_priamide
        # 196
        self_feats[i, 195] = mol_graph.fr_prisulfonamd
        self_feats[i, 196] = mol_graph.fr_pyridine
        self_feats[i, 197] = mol_graph.fr_quatN
        self_feats[i, 198] = mol_graph.fr_sulfide
        self_feats[i, 199] = mol_graph.fr_sulfonamd
        # 201
        self_feats[i, 200] = mol_graph.fr_sulfone
        self_feats[i, 201] = mol_graph.fr_term_acetylene
        self_feats[i, 202] = mol_graph.fr_tetrazole
        self_feats[i, 203] = mol_graph.fr_thiazole
        self_feats[i, 204] = mol_graph.fr_thiocyan
        # 206
        self_feats[i, 205] = mol_graph.fr_thiophene
        self_feats[i, 206] = mol_graph.fr_unbrch_alkane
        self_feats[i, 207] = mol_graph.fr_urea
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# freesolv
def read_dataset_freesolv_bilinear(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            # 1
            mol_graph.MaxEStateIndex = dsc.MaxEStateIndex(mol)
            mol_graph.MinEStateIndex = dsc.MinEStateIndex(mol)
            mol_graph.MaxAbsEStateIndex = dsc.MaxAbsEStateIndex(mol)
            mol_graph.MinAbsEStateIndex = dsc.MinAbsEStateIndex(mol)
            mol_graph.qed = dsc.qed(mol)
            # 6
            mol_graph.MolWt = dsc.MolWt(mol)
            mol_graph.HeavyAtomMolWt = dsc.HeavyAtomMolWt(mol)
            mol_graph.ExactMolWt = dsc.ExactMolWt(mol)
            mol_graph.NumValenceElectrons = dsc.NumValenceElectrons(mol)
            mol_graph.NumRadicalElectrons = dsc.NumRadicalElectrons(mol)
            # 11
            mol_graph.MaxPartialCharge = dsc.MaxPartialCharge(mol)
            mol_graph.MinPartialCharge = dsc.MinPartialCharge(mol)
            mol_graph.MaxAbsPartialCharge = dsc.MaxAbsPartialCharge(mol)
            mol_graph.MinAbsPartialCharge = dsc.MinAbsPartialCharge(mol)
            mol_graph.FpDensityMorgan1 = dsc.FpDensityMorgan1(mol)
            # 16
            mol_graph.FpDensityMorgan2 = dsc.FpDensityMorgan2(mol)
            mol_graph.FpDensityMorgan3 = dsc.FpDensityMorgan3(mol)
            mol_graph.BCUT2D_MWHI = dsc.BCUT2D_MWHI(mol)
            mol_graph.BCUT2D_MWLOW = dsc.BCUT2D_MWLOW(mol)
            mol_graph.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(mol)
            # 21
            mol_graph.BCUT2D_CHGLO = dsc.BCUT2D_CHGLO(mol)
            mol_graph.BCUT2D_LOGPHI = dsc.BCUT2D_LOGPHI(mol)
            mol_graph.BCUT2D_LOGPLOW = dsc.BCUT2D_LOGPLOW(mol)
            mol_graph.BCUT2D_MRHI = dsc.BCUT2D_MRHI(mol)
            mol_graph.BCUT2D_MRLOW = dsc.BCUT2D_MRLOW(mol)
            # 26
            mol_graph.BalabanJ = dsc.BalabanJ(mol)
            mol_graph.BertzCT = dsc.BertzCT(mol)
            mol_graph.Chi0 = dsc.Chi0(mol)
            mol_graph.Chi0n = dsc.Chi0n(mol)
            mol_graph.Chi0v = dsc.Chi0v(mol)
            # 31
            mol_graph.Chi1 = dsc.Chi1(mol)
            mol_graph.Chi1n = dsc.Chi1n(mol)
            mol_graph.Chi1v = dsc.Chi1v(mol)
            mol_graph.Chi2n = dsc.Chi2n(mol)
            mol_graph.Chi2v = dsc.Chi2v(mol)
            # 36
            mol_graph.Chi3n = dsc.Chi3n(mol)
            mol_graph.Chi3v = dsc.Chi3v(mol)
            mol_graph.Chi4n = dsc.Chi4n(mol)
            mol_graph.Chi4v = dsc.Chi4v(mol)
            mol_graph.HallKierAlpha = dsc.HallKierAlpha(mol)
            # 41
            mol_graph.Ipc = dsc.Ipc(mol)
            mol_graph.Kappa1 = dsc.Kappa1(mol)
            mol_graph.Kappa2 = dsc.Kappa2(mol)
            mol_graph.Kappa3 = dsc.Kappa3(mol)
            mol_graph.LabuteASA = dsc.LabuteASA(mol)
            # 46
            mol_graph.PEOE_VSA1 = dsc.PEOE_VSA1(mol)
            mol_graph.PEOE_VSA10 = dsc.PEOE_VSA10(mol)
            mol_graph.PEOE_VSA11 = dsc.PEOE_VSA11(mol)
            mol_graph.PEOE_VSA12 = dsc.PEOE_VSA12(mol)
            mol_graph.PEOE_VSA13 = dsc.PEOE_VSA13(mol)
            # 51
            mol_graph.PEOE_VSA14 = dsc.PEOE_VSA14(mol)
            mol_graph.PEOE_VSA2 = dsc.PEOE_VSA2(mol)
            mol_graph.PEOE_VSA3 = dsc.PEOE_VSA3(mol)
            mol_graph.PEOE_VSA4 = dsc.PEOE_VSA4(mol)
            mol_graph.PEOE_VSA5 = dsc.PEOE_VSA5(mol)
            # 56
            mol_graph.PEOE_VSA6 = dsc.PEOE_VSA6(mol)
            mol_graph.PEOE_VSA7 = dsc.PEOE_VSA7(mol)
            mol_graph.PEOE_VSA8 = dsc.PEOE_VSA8(mol)
            mol_graph.PEOE_VSA9 = dsc.PEOE_VSA9(mol)
            mol_graph.SMR_VSA1 = dsc.SMR_VSA1(mol)
            # 61
            mol_graph.SMR_VSA10 = dsc.SMR_VSA10(mol)
            mol_graph.SMR_VSA2 = dsc.SMR_VSA2(mol)
            mol_graph.SMR_VSA3 = dsc.SMR_VSA3(mol)
            mol_graph.SMR_VSA4 = dsc.SMR_VSA4(mol)
            mol_graph.SMR_VSA5 = dsc.SMR_VSA5(mol)
            # 66
            mol_graph.SMR_VSA6 = dsc.SMR_VSA6(mol)
            mol_graph.SMR_VSA7 = dsc.SMR_VSA7(mol)
            mol_graph.SMR_VSA8 = dsc.SMR_VSA8(mol)
            mol_graph.SMR_VSA9 = dsc.SMR_VSA9(mol)
            mol_graph.SlogP_VSA1 = dsc.SlogP_VSA1(mol)
            # 71
            mol_graph.SlogP_VSA10 = dsc.SlogP_VSA10(mol)
            mol_graph.SlogP_VSA11 = dsc.SlogP_VSA11(mol)
            mol_graph.SlogP_VSA12 = dsc.SlogP_VSA12(mol)
            mol_graph.SlogP_VSA2 = dsc.SlogP_VSA2(mol)
            mol_graph.SlogP_VSA3 = dsc.SlogP_VSA3(mol)
            # 76
            mol_graph.SlogP_VSA4 = dsc.SlogP_VSA4(mol)
            mol_graph.SlogP_VSA5 = dsc.SlogP_VSA5(mol)
            mol_graph.SlogP_VSA6 = dsc.SlogP_VSA6(mol)
            mol_graph.SlogP_VSA7 = dsc.SlogP_VSA7(mol)
            mol_graph.SlogP_VSA8 = dsc.SlogP_VSA8(mol)
            # 81
            mol_graph.SlogP_VSA9 = dsc.SlogP_VSA9(mol)
            mol_graph.TPSA = dsc.TPSA(mol)
            mol_graph.EState_VSA1 = dsc.EState_VSA1(mol)
            mol_graph.EState_VSA10 = dsc.EState_VSA10(mol)
            mol_graph.EState_VSA11 = dsc.EState_VSA11(mol)
            # 86
            mol_graph.EState_VSA2 = dsc.EState_VSA2(mol)
            mol_graph.EState_VSA3 = dsc.EState_VSA3(mol)
            mol_graph.EState_VSA4 = dsc.EState_VSA4(mol)
            mol_graph.EState_VSA5 = dsc.EState_VSA5(mol)
            mol_graph.EState_VSA6 = dsc.EState_VSA6(mol)
            # 91
            mol_graph.EState_VSA7 = dsc.EState_VSA7(mol)
            mol_graph.EState_VSA8 = dsc.EState_VSA8(mol)
            mol_graph.EState_VSA9 = dsc.EState_VSA9(mol)
            mol_graph.VSA_EState1 = dsc.VSA_EState1(mol)
            mol_graph.VSA_EState10 = dsc.VSA_EState10(mol)
            # 96
            mol_graph.VSA_EState2 = dsc.VSA_EState2(mol)
            mol_graph.VSA_EState3 = dsc.VSA_EState3(mol)
            mol_graph.VSA_EState4 = dsc.VSA_EState4(mol)
            mol_graph.VSA_EState5 = dsc.VSA_EState5(mol)
            mol_graph.VSA_EState6 = dsc.VSA_EState6(mol)
            # 101
            mol_graph.VSA_EState7 = dsc.VSA_EState7(mol)
            mol_graph.VSA_EState8 = dsc.VSA_EState8(mol)
            mol_graph.VSA_EState9 = dsc.VSA_EState9(mol)
            mol_graph.FractionCSP3 = dsc.FractionCSP3(mol)
            mol_graph.HeavyAtomCount = dsc.HeavyAtomCount(mol)
            # 106
            mol_graph.NHOHCount = dsc.NHOHCount(mol)
            mol_graph.NOCount = dsc.NOCount(mol)
            mol_graph.NumAliphaticCarbocycles = dsc.NumAliphaticCarbocycles(mol)
            mol_graph.NumAliphaticHeterocycles = dsc.NumAliphaticHeterocycles(mol)
            mol_graph.NumAliphaticRings = dsc.NumAliphaticRings(mol)
            # 111
            mol_graph.NumAromaticCarbocycles = dsc.NumAromaticCarbocycles(mol)
            mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            mol_graph.NumAromaticRings = dsc.NumAromaticRings(mol)
            mol_graph.NumHAcceptors = dsc.NumHAcceptors(mol)
            mol_graph.NumHDonors = dsc.NumHDonors(mol)
            # 116
            mol_graph.NumHeteroatoms = dsc.NumHeteroatoms(mol)
            mol_graph.NumRotatableBonds = dsc.NumRotatableBonds(mol)
            mol_graph.NumSaturatedCarbocycles = dsc.NumSaturatedCarbocycles(mol)
            mol_graph.NumSaturatedHeterocycles = dsc.NumSaturatedHeterocycles(mol)
            mol_graph.NumSaturatedRings = dsc.NumSaturatedRings(mol)
            # 121
            mol_graph.RingCount = dsc.RingCount(mol)
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.MolMR = dsc.MolMR(mol)
            mol_graph.fr_Al_COO = dsc.fr_Al_COO(mol)
            mol_graph.fr_Al_OH = dsc.fr_Al_OH(mol)
            # 126
            mol_graph.fr_Al_OH_noTert = dsc.fr_Al_OH_noTert(mol)
            mol_graph.fr_ArN = dsc.fr_ArN(mol)
            mol_graph.fr_Ar_COO = dsc.fr_Ar_COO(mol)
            mol_graph.fr_Ar_N = dsc.fr_Ar_N(mol)
            mol_graph.fr_Ar_NH = dsc.fr_Ar_NH(mol)
            # 131
            mol_graph.fr_Ar_OH = dsc.fr_Ar_OH(mol)
            mol_graph.fr_COO = dsc.fr_COO(mol)
            mol_graph.fr_COO2 = dsc.fr_COO2(mol)
            mol_graph.fr_C_O = dsc.fr_C_O(mol)
            mol_graph.fr_C_O_noCOO = dsc.fr_C_O_noCOO(mol)
            # 136
            mol_graph.fr_C_S = dsc.fr_C_S(mol)
            mol_graph.fr_HOCCN = dsc.fr_HOCCN(mol)
            mol_graph.fr_Imine = dsc.fr_Imine(mol)
            mol_graph.fr_NH0 = dsc.fr_NH0(mol)
            mol_graph.fr_NH1 = dsc.fr_NH1(mol)
            # 141
            mol_graph.fr_NH2 = dsc.fr_NH2(mol)
            mol_graph.fr_N_O = dsc.fr_N_O(mol)
            mol_graph.fr_Ndealkylation1 = dsc.fr_Ndealkylation1(mol)
            mol_graph.fr_Ndealkylation2 = dsc.fr_Ndealkylation2(mol)
            mol_graph.fr_Nhpyrrole = dsc.fr_Nhpyrrole(mol)
            # 146
            mol_graph.fr_SH = dsc.fr_SH(mol)
            mol_graph.fr_aldehyde = dsc.fr_aldehyde(mol)
            mol_graph.fr_alkyl_carbamate = dsc.fr_alkyl_carbamate(mol)
            mol_graph.fr_alkyl_halide = dsc.fr_alkyl_halide(mol)
            mol_graph.fr_allylic_oxid = dsc.fr_allylic_oxid(mol)
            # 151
            mol_graph.fr_amide = dsc.fr_amide(mol)
            mol_graph.fr_amidine = dsc.fr_amidine(mol)
            mol_graph.fr_aniline = dsc.fr_aniline(mol)
            mol_graph.fr_aryl_methyl = dsc.fr_aryl_methyl(mol)
            mol_graph.fr_azide = dsc.fr_azide(mol)
            # 156
            mol_graph.fr_azo = dsc.fr_azo(mol)
            mol_graph.fr_barbitur = dsc.fr_barbitur(mol)
            mol_graph.fr_benzene = dsc.fr_benzene(mol)
            mol_graph.fr_benzodiazepine = dsc.fr_benzodiazepine(mol)
            mol_graph.fr_bicyclic = dsc.fr_bicyclic(mol)
            # 161
            mol_graph.fr_diazo = dsc.fr_diazo(mol)
            mol_graph.fr_dihydropyridine = dsc.fr_dihydropyridine(mol)
            mol_graph.fr_epoxide = dsc.fr_epoxide(mol)
            mol_graph.fr_ester = dsc.fr_ester(mol)
            mol_graph.fr_ether = dsc.fr_ether(mol)
            # 166
            mol_graph.fr_furan = dsc.fr_furan(mol)
            mol_graph.fr_guanido = dsc.fr_guanido(mol)
            mol_graph.fr_halogen = dsc.fr_halogen(mol)
            mol_graph.fr_hdrzine = dsc.fr_hdrzine(mol)
            mol_graph.fr_hdrzone = dsc.fr_hdrzone(mol)
            # 171
            mol_graph.fr_imidazole = dsc.fr_imidazole(mol)
            mol_graph.fr_imide = dsc.fr_imide(mol)
            mol_graph.fr_isocyan = dsc.fr_isocyan(mol)
            mol_graph.fr_isothiocyan = dsc.fr_isothiocyan(mol)
            mol_graph.fr_ketone = dsc.fr_ketone(mol)
            # 176
            mol_graph.fr_ketone_Topliss = dsc.fr_ketone_Topliss(mol)
            mol_graph.fr_lactam = dsc.fr_lactam(mol)
            mol_graph.fr_lactone = dsc.fr_lactone(mol)
            mol_graph.fr_methoxy = dsc.fr_methoxy(mol)
            mol_graph.fr_morpholine = dsc.fr_morpholine(mol)
            # 181
            mol_graph.fr_nitrile = dsc.fr_nitrile(mol)
            mol_graph.fr_nitro = dsc.fr_nitro(mol)
            mol_graph.fr_nitro_arom = dsc.fr_nitro_arom(mol)
            mol_graph.fr_nitro_arom_nonortho = dsc.fr_nitro_arom_nonortho(mol)
            mol_graph.fr_nitroso = dsc.fr_nitroso(mol)
            # 186
            mol_graph.fr_oxazole = dsc.fr_oxazole(mol)
            mol_graph.fr_oxime = dsc.fr_oxime(mol)
            mol_graph.fr_para_hydroxylation = dsc.fr_para_hydroxylation(mol)
            mol_graph.fr_phenol = dsc.fr_phenol(mol)
            mol_graph.fr_phenol_noOrthoHbond = dsc.fr_phenol_noOrthoHbond(mol)
            # 191
            mol_graph.fr_phos_acid = dsc.fr_phos_acid(mol)
            mol_graph.fr_phos_ester = dsc.fr_phos_ester(mol)
            mol_graph.fr_piperdine = dsc.fr_piperdine(mol)
            mol_graph.fr_piperzine = dsc.fr_piperzine(mol)
            mol_graph.fr_priamide = dsc.fr_priamide(mol)
            # 196
            mol_graph.fr_prisulfonamd = dsc.fr_prisulfonamd(mol)
            mol_graph.fr_pyridine = dsc.fr_pyridine(mol)
            mol_graph.fr_quatN = dsc.fr_quatN(mol)
            mol_graph.fr_sulfide = dsc.fr_sulfide(mol)
            mol_graph.fr_sulfonamd = dsc.fr_sulfonamd(mol)
            # 201
            mol_graph.fr_sulfone = dsc.fr_sulfone(mol)
            mol_graph.fr_term_acetylene = dsc.fr_term_acetylene(mol)
            mol_graph.fr_tetrazole = dsc.fr_tetrazole(mol)
            mol_graph.fr_thiazole = dsc.fr_thiazole(mol)
            mol_graph.fr_thiocyan = dsc.fr_thiocyan(mol)
            # 206
            mol_graph.fr_thiophene = dsc.fr_thiophene(mol)
            mol_graph.fr_unbrch_alkane = dsc.fr_unbrch_alkane(mol)
            mol_graph.fr_urea = dsc.fr_urea(mol)

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    for feat in ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']:
        FeatureNormalization(mol_graphs, feat)

    return samples

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

# Solubility
def descriptor_selection_solubility(samples):
    self_feats = np.empty((len(samples), 16), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 1
        self_feats[i, 0] = mol_graph.Chi1
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.MolLogP
        self_feats[i, 3] = mol_graph.PEOE_VSA6
        self_feats[i, 4] = mol_graph.VSA_EState6
        # 6
        self_feats[i, 5] = mol_graph.SMR_VSA10
        self_feats[i, 6] = mol_graph.Kappa1
        self_feats[i, 7] = mol_graph.fr_benzene
        self_feats[i, 8] = mol_graph.fr_quatN
        self_feats[i, 9] = mol_graph.SlogP_VSA6
        # 11
        self_feats[i, 10] = mol_graph.NumHDonors
        self_feats[i, 11] = mol_graph.EState_VSA2
        self_feats[i, 12] = mol_graph.PEOE_VSA7
        self_feats[i, 13] = mol_graph.FpDensityMorgan1
        self_feats[i, 14] = mol_graph.NumAliphaticCarbocycles
        # 16
        self_feats[i, 15] = mol_graph.TPSA

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
