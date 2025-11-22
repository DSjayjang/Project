import numpy as np
import pandas as pd
import random
import torch
import rdkit.Chem.Descriptors as dsc

from utils.utils import FeatureNormalization
from utils.mol_graph import smiles_to_mol_graph

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_scaffold_groups(smiles_list):
    scaffold_dict = {}

    for idx, s in enumerate(smiles_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s)

        if scaffold not in scaffold_dict:
            scaffold_dict[scaffold] = []

        scaffold_dict[scaffold].append(idx)

    return scaffold_dict


def scaffold_kfold_split(samples, K=5):
    scaffold_groups = get_scaffold_groups(samples)
    print('scaffold_groups:', scaffold_groups)

    groups = list(scaffold_groups.values())

    random.shuffle(groups)

    folds = [[] for _ in range(K)]

    for i, group in enumerate(groups):
        folds[i % K].extend(group)

    return folds

def read_dataset(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    smiles_list = []

    target = np.array(data_mat[:, 1:3], dtype=float)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            mol_graph.num_atoms = mol.GetNumAtoms()
            mol_graph.weight = dsc.ExactMolWt(mol)
            mol_graph.num_rings = mol.GetRingInfo().NumRings()

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)
            smiles_list.append(smiles[i])

    for feat in ['num_atoms', 'weight', 'num_rings']:
        FeatureNormalization(mol_graphs, feat)

    return samples, smiles_list


# freesolv
def read_dataset_freesolv(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)
    smiles_list = []

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
            smiles_list.append(smiles[i])

    for feat in ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']:
        FeatureNormalization(mol_graphs, feat)

    return samples, smiles_list


# esol
def read_dataset_esol(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)
    smiles_list = []

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            # 1
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.MaxAbsPartialCharge = dsc.MaxAbsPartialCharge(mol)
            mol_graph.MaxEStateIndex = dsc.MaxEStateIndex(mol)
            mol_graph.SMR_VSA10 = dsc.SMR_VSA10(mol)
            mol_graph.Kappa2 = dsc.Kappa2(mol)
            # 6
            mol_graph.BCUT2D_MWLOW = dsc.BCUT2D_MWLOW(mol)
            mol_graph.PEOE_VSA13 = dsc.PEOE_VSA13(mol)
            mol_graph.MinAbsPartialCharge = dsc.MinAbsPartialCharge(mol)
            mol_graph.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(mol)
            mol_graph.PEOE_VSA6 = dsc.PEOE_VSA6(mol)
            # 11
            mol_graph.SlogP_VSA1 = dsc.SlogP_VSA1(mol)
            mol_graph.fr_nitro = dsc.fr_nitro(mol)
            mol_graph.BalabanJ = dsc.BalabanJ(mol)
            mol_graph.SMR_VSA9 = dsc.SMR_VSA9(mol)
            mol_graph.fr_alkyl_halide = dsc.fr_alkyl_halide(mol)
            # 16
            mol_graph.fr_hdrzine = dsc.fr_hdrzine(mol)
            mol_graph.PEOE_VSA8 = dsc.PEOE_VSA8(mol)
            mol_graph.fr_Ar_NH = dsc.fr_Ar_NH(mol)
            mol_graph.fr_imidazole = dsc.fr_imidazole(mol)
            mol_graph.fr_Nhpyrrole = dsc.fr_Nhpyrrole(mol)
            # 21
            mol_graph.EState_VSA5 = dsc.EState_VSA5(mol)
            mol_graph.PEOE_VSA4 = dsc.PEOE_VSA4(mol)
            mol_graph.fr_ester = dsc.fr_ester(mol)
            mol_graph.PEOE_VSA2 = dsc.PEOE_VSA2(mol)
            mol_graph.NumAromaticCarbocycles = dsc.NumAromaticCarbocycles(mol)
            # 26
            mol_graph.BCUT2D_LOGPHI = dsc.BCUT2D_LOGPHI(mol)
            mol_graph.EState_VSA11 = dsc.EState_VSA11(mol)
            mol_graph.fr_furan = dsc.fr_furan(mol)
            mol_graph.EState_VSA2 = dsc.EState_VSA2(mol)
            mol_graph.fr_benzene = dsc.fr_benzene(mol)
            # 31
            mol_graph.fr_sulfide = dsc.fr_sulfide(mol)
            mol_graph.fr_aryl_methyl = dsc.fr_aryl_methyl(mol)
            mol_graph.SlogP_VSA10 = dsc.SlogP_VSA10(mol)
            mol_graph.HeavyAtomMolWt = dsc.HeavyAtomMolWt(mol)
            mol_graph.fr_nitro_arom_nonortho = dsc.fr_nitro_arom_nonortho(mol)
            # 36
            mol_graph.FpDensityMorgan2 = dsc.FpDensityMorgan2(mol)
            mol_graph.EState_VSA8 = dsc.EState_VSA8(mol)
            mol_graph.fr_bicyclic = dsc.fr_bicyclic(mol)
            mol_graph.fr_aniline = dsc.fr_aniline(mol)
            mol_graph.fr_allylic_oxid = dsc.fr_allylic_oxid(mol)
            # 41
            mol_graph.fr_C_S = dsc.fr_C_S(mol)
            mol_graph.SlogP_VSA7 = dsc.SlogP_VSA7(mol)
            mol_graph.SlogP_VSA4 = dsc.SlogP_VSA4(mol)
            mol_graph.fr_para_hydroxylation = dsc.fr_para_hydroxylation(mol)
            mol_graph.PEOE_VSA7 = dsc.PEOE_VSA7(mol)
            # 46
            mol_graph.fr_Al_OH_noTert = dsc.fr_Al_OH_noTert(mol)
            mol_graph.fr_pyridine = dsc.fr_pyridine(mol)
            mol_graph.fr_phos_acid = dsc.fr_phos_acid(mol)
            mol_graph.fr_phos_ester = dsc.fr_phos_ester(mol)
            mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            # 51
            mol_graph.EState_VSA7 = dsc.EState_VSA7(mol)
            mol_graph.PEOE_VSA12 = dsc.PEOE_VSA12(mol)
            mol_graph.Ipc = dsc.Ipc(mol)
            mol_graph.FpDensityMorgan1 = dsc.FpDensityMorgan1(mol)
            mol_graph.PEOE_VSA14 = dsc.PEOE_VSA14(mol)
            # 56
            mol_graph.fr_guanido = dsc.fr_guanido(mol)
            mol_graph.fr_benzodiazepine = dsc.fr_benzodiazepine(mol)
            mol_graph.fr_thiophene = dsc.fr_thiophene(mol)
            mol_graph.fr_Ndealkylation1 = dsc.fr_Ndealkylation1(mol)
            mol_graph.fr_aldehyde = dsc.fr_aldehyde(mol)
            # 61
            mol_graph.fr_term_acetylene = dsc.fr_term_acetylene(mol)
            mol_graph.SMR_VSA2 = dsc.SMR_VSA2(mol)
            mol_graph.fr_lactone = dsc.fr_lactone(mol)

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)
            smiles_list.append(smiles[i])

    for feat in ['MolLogP', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'SMR_VSA10', 'Kappa2', 
                'BCUT2D_MWLOW', 'PEOE_VSA13', 'MinAbsPartialCharge', 'BCUT2D_CHGHI', 'PEOE_VSA6', 
                'SlogP_VSA1', 'fr_nitro', 'BalabanJ', 'SMR_VSA9', 'fr_alkyl_halide', 
                'fr_hdrzine', 'PEOE_VSA8', 'fr_Ar_NH', 'fr_imidazole', 'fr_Nhpyrrole', 
                'EState_VSA5', 'PEOE_VSA4', 'fr_ester', 'PEOE_VSA2', 'NumAromaticCarbocycles', 
                'BCUT2D_LOGPHI', 'EState_VSA11', 'fr_furan', 'EState_VSA2', 'fr_benzene', 
                'fr_sulfide', 'fr_aryl_methyl', 'SlogP_VSA10', 'HeavyAtomMolWt', 'fr_nitro_arom_nonortho', 
                'FpDensityMorgan2', 'EState_VSA8', 'fr_bicyclic', 'fr_aniline', 'fr_allylic_oxid', 
                'fr_C_S', 'SlogP_VSA7', 'SlogP_VSA4', 'fr_para_hydroxylation', 'PEOE_VSA7', 
                'fr_Al_OH_noTert', 'fr_pyridine', 'fr_phos_acid', 'fr_phos_ester', 'NumAromaticHeterocycles', 
                'EState_VSA7', 'PEOE_VSA12', 'Ipc', 'FpDensityMorgan1', 'PEOE_VSA14', 
                'fr_guanido', 'fr_benzodiazepine', 'fr_thiophene', 'fr_Ndealkylation1', 'fr_aldehyde', 
                'fr_term_acetylene', 'SMR_VSA2', 'fr_lactone']:
        FeatureNormalization(mol_graphs, feat)

    return samples, smiles_list

# lipo
def read_dataset_lipo(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)
    smiles_list = []

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            # 1
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.fr_COO = dsc.fr_COO(mol)
            mol_graph.Ipc = dsc.Ipc(mol)
            mol_graph.fr_sulfonamd = dsc.fr_sulfonamd(mol)
            mol_graph.PEOE_VSA7 = dsc.PEOE_VSA7(mol)
            # 6
            mol_graph.PEOE_VSA13 = dsc.PEOE_VSA13(mol)
            mol_graph.SlogP_VSA10 = dsc.SlogP_VSA10(mol)
            mol_graph.fr_unbrch_alkane = dsc.fr_unbrch_alkane(mol)
            mol_graph.SMR_VSA10 = dsc.SMR_VSA10(mol)
            mol_graph.PEOE_VSA12 = dsc.PEOE_VSA12(mol)
            # 11
            mol_graph.fr_guanido = dsc.fr_guanido(mol)
            mol_graph.FpDensityMorgan1 = dsc.FpDensityMorgan1(mol)
            mol_graph.NHOHCount = dsc.NHOHCount(mol)
            mol_graph.fr_sulfide = dsc.fr_sulfide(mol)
            mol_graph.VSA_EState5 = dsc.VSA_EState5(mol)
            # 16
            mol_graph.fr_HOCCN = dsc.fr_HOCCN(mol)
            mol_graph.fr_piperdine = dsc.fr_piperdine(mol)
            mol_graph.NumSaturatedCarbocycles = dsc.NumSaturatedCarbocycles(mol)
            mol_graph.fr_amidine = dsc.fr_amidine(mol)
            mol_graph.NumHDonors = dsc.NumHDonors(mol)
            # 21
            mol_graph.NumAromaticRings = dsc.NumAromaticRings(mol)
            mol_graph.BalabanJ = dsc.BalabanJ(mol)
            mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            mol_graph.MinEStateIndex = dsc.MinEStateIndex(mol)
            mol_graph.fr_Ar_N = dsc.fr_Ar_N(mol)

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)
            smiles_list.append(smiles[i])

    for feat in ['MolLogP', 'fr_COO', 'Ipc', 'fr_sulfonamd', 'PEOE_VSA7',
                'PEOE_VSA13', 'SlogP_VSA10', 'fr_unbrch_alkane', 'SMR_VSA10', 'PEOE_VSA12',
                'fr_guanido', 'FpDensityMorgan1', 'NHOHCount', 'fr_sulfide', 'VSA_EState5',
                'fr_HOCCN', 'fr_piperdine', 'NumSaturatedCarbocycles', 'fr_amidine', 'NumHDonors',
                'NumAromaticRings', 'BalabanJ', 'NumAromaticHeterocycles', 'MinEStateIndex', 'fr_Ar_N']:
        FeatureNormalization(mol_graphs, feat)

    return samples, smiles_list


# Self-Curated Gas
def read_dataset_scgas(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)
    smiles_list = []

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            # 1
            mol_graph.MolMR = dsc.MolMR(mol)
            mol_graph.TPSA = dsc.TPSA(mol)
            mol_graph.fr_halogen = dsc.fr_halogen(mol)
            mol_graph.SlogP_VSA12 = dsc.SlogP_VSA12(mol)
            mol_graph.RingCount = dsc.RingCount(mol)
            # 6
            mol_graph.Kappa1 = dsc.Kappa1(mol)
            mol_graph.NumHAcceptors = dsc.NumHAcceptors(mol)
            mol_graph.NumHDonors = dsc.NumHDonors(mol)
            mol_graph.SMR_VSA7 = dsc.SMR_VSA7(mol)
            mol_graph.SMR_VSA5 = dsc.SMR_VSA5(mol)
            # 11
            mol_graph.Chi1 = dsc.Chi1(mol)
            mol_graph.Chi3n = dsc.Chi3n(mol)
            mol_graph.BertzCT = dsc.BertzCT(mol)
            mol_graph.VSA_EState8 = dsc.VSA_EState8(mol)
            mol_graph.NumAliphaticCarbocycles = dsc.NumAliphaticCarbocycles(mol)
            # 16
            mol_graph.HallKierAlpha = dsc.HallKierAlpha(mol)
            mol_graph.VSA_EState6 = dsc.VSA_EState6(mol)
            mol_graph.NumAromaticRings = dsc.NumAromaticRings(mol)
            mol_graph.Chi4n = dsc.Chi4n(mol)
            mol_graph.PEOE_VSA7 = dsc.PEOE_VSA7(mol)
            # 21
            mol_graph.SlogP_VSA5 = dsc.SlogP_VSA5(mol)
            mol_graph.VSA_EState7 = dsc.VSA_EState7(mol)
            mol_graph.NOCount = dsc.NOCount(mol)

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)
            smiles_list.append(smiles[i])

    for feat in ['MolMR', 'TPSA', 'fr_halogen', 'SlogP_VSA12', 'RingCount', 
                'Kappa1', 'NumHAcceptors', 'NumHDonors', 'SMR_VSA7', 'SMR_VSA5',
                'Chi1', 'Chi3n', 'BertzCT', 'VSA_EState8', 'NumAliphaticCarbocycles', 
                'HallKierAlpha', 'VSA_EState6', 'NumAromaticRings', 'Chi4n', 'PEOE_VSA7', 
                'SlogP_VSA5', 'VSA_EState7', 'NOCount']:
        FeatureNormalization(mol_graphs, feat)

    return samples, smiles_list


# Solubility
def read_dataset_solubility(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)
    smiles_list = []

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            # 1
            mol_graph.Chi1v = dsc.Chi1v(mol)
            mol_graph.Chi1 = dsc.Chi1(mol)
            mol_graph.SlogP_VSA2 = dsc.SlogP_VSA2(mol)
            mol_graph.HallKierAlpha = dsc.HallKierAlpha(mol)
            mol_graph.PEOE_VSA6 = dsc.PEOE_VSA6(mol)
            # 6
            mol_graph.fr_benzene = dsc.fr_benzene(mol)
            mol_graph.BertzCT = dsc.BertzCT(mol)
            mol_graph.VSA_EState6 = dsc.VSA_EState6(mol)
            mol_graph.SMR_VSA7 = dsc.SMR_VSA7(mol)
            mol_graph.Chi3n = dsc.Chi3n(mol)
            # 11
            mol_graph.HeavyAtomMolWt = dsc.HeavyAtomMolWt(mol)
            mol_graph.SMR_VSA10 = dsc.SMR_VSA10(mol)
            mol_graph.Kappa1 = dsc.Kappa1(mol)
            mol_graph.fr_quatN = dsc.fr_quatN(mol)
            mol_graph.PEOE_VSA7 = dsc.PEOE_VSA7(mol)
            # 16
            mol_graph.NumHDonors = dsc.NumHDonors(mol)
            mol_graph.MinEStateIndex = dsc.MinEStateIndex(mol)
            mol_graph.fr_C_O_noCOO = dsc.fr_C_O_noCOO(mol)
            mol_graph.EState_VSA1 = dsc.EState_VSA1(mol)
            mol_graph.MolLogP = dsc.MolLogP(mol)
            # 21
            mol_graph.fr_halogen = dsc.fr_halogen(mol)
            mol_graph.SlogP_VSA3 = dsc.SlogP_VSA3(mol)
            mol_graph.SlogP_VSA5 = dsc.SlogP_VSA5(mol)
            mol_graph.SlogP_VSA1 = dsc.SlogP_VSA1(mol)
            mol_graph.SlogP_VSA12 = dsc.SlogP_VSA12(mol)
            # 26
            mol_graph.VSA_EState10 = dsc.VSA_EState10(mol)
            mol_graph.MinPartialCharge = dsc.MinPartialCharge(mol)
            mol_graph.Kappa2 = dsc.Kappa2(mol)
            mol_graph.NHOHCount = dsc.NHOHCount(mol)
            mol_graph.SlogP_VSA6 = dsc.SlogP_VSA6(mol)

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)
            smiles_list.append(smiles[i])

    for feat in ['Chi1v', 'Chi1', 'SlogP_VSA2', 'HallKierAlpha', 'PEOE_VSA6',
       'fr_benzene', 'BertzCT', 'VSA_EState6', 'SMR_VSA7', 'Chi3n',
       'HeavyAtomMolWt', 'SMR_VSA10', 'Kappa1', 'fr_quatN', 'PEOE_VSA7',
       'NumHDonors', 'MinEStateIndex', 'fr_C_O_noCOO', 'EState_VSA1',
       'MolLogP', 'fr_halogen', 'SlogP_VSA3', 'SlogP_VSA5', 'SlogP_VSA1',
       'SlogP_VSA12', 'VSA_EState10', 'MinPartialCharge', 'Kappa2',
       'NHOHCount', 'SlogP_VSA6']:
        FeatureNormalization(mol_graphs, feat)

    return samples, smiles_list