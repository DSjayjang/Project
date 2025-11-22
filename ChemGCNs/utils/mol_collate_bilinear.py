import numpy as np
import torch

import dgl
import pandas as pd
import rdkit.Chem.Descriptors as dsc

from utils.utils import FeatureNormalization
from utils.mol_graph import smiles_to_mol_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Freesolv
def descriptor_selection_freesolv(samples):
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


# ESOL
def descriptor_selection_esol(samples):
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

# Lipo
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

# Solubility
def descriptor_selection_solubility(samples):
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
