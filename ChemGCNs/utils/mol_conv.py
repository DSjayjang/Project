import numpy as np
import pandas as pd
import torch
import rdkit.Chem.Descriptors as dsc

from utils.utils import FeatureNormalization
from utils.mol_graph import smiles_to_mol_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_dataset(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]

    target = np.array(data_mat[:, 1:3], dtype=float)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            mol_graph.num_atoms = mol.GetNumAtoms()
            mol_graph.weight = dsc.ExactMolWt(mol)
            mol_graph.num_rings = mol.GetRingInfo().NumRings()

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    for feat in ['num_atoms', 'weight', 'num_rings']:
        FeatureNormalization(mol_graphs, feat)

    return samples

# freesolv
def read_dataset_freesolv(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            # 1
            mol_graph.NHOHCount = dsc.NHOHCount(mol)
            mol_graph.SlogP_VSA2 = dsc.SlogP_VSA2(mol)
            mol_graph.SlogP_VSA10 = dsc.SlogP_VSA10(mol)
            mol_graph.NumAromaticRings = dsc.NumAromaticRings(mol)
            mol_graph.MaxEStateIndex = dsc.MaxEStateIndex(mol)
            # 6
            mol_graph.PEOE_VSA14 = dsc.PEOE_VSA14(mol)
            mol_graph.fr_Ar_NH = dsc.fr_Ar_NH(mol)
            mol_graph.SMR_VSA3 = dsc.SMR_VSA3(mol)
            mol_graph.SMR_VSA7 = dsc.SMR_VSA7(mol)
            mol_graph.SlogP_VSA5 = dsc.SlogP_VSA5(mol)
            # 11
            mol_graph.VSA_EState8 = dsc.VSA_EState8(mol)
            mol_graph.MaxAbsEStateIndex = dsc.MaxAbsEStateIndex(mol)
            mol_graph.PEOE_VSA2 = dsc.PEOE_VSA2(mol)
            mol_graph.fr_Nhpyrrole = dsc.fr_Nhpyrrole(mol)
            mol_graph.fr_amide = dsc.fr_amide(mol)
            # 16
            mol_graph.SlogP_VSA3 = dsc.SlogP_VSA3(mol)
            mol_graph.BCUT2D_MRHI = dsc.BCUT2D_MRHI(mol)
            mol_graph.fr_nitrile = dsc.fr_nitrile(mol)
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.PEOE_VSA10 = dsc.PEOE_VSA10(mol)
            # 21
            mol_graph.MinPartialCharge = dsc.MinPartialCharge(mol)
            mol_graph.fr_Al_OH = dsc.fr_Al_OH(mol)
            mol_graph.fr_sulfone = dsc.fr_sulfone(mol)
            mol_graph.fr_Al_COO = dsc.fr_Al_COO(mol)
            mol_graph.fr_nitro_arom_nonortho = dsc.fr_nitro_arom_nonortho(mol)
            # 26
            mol_graph.fr_imidazole = dsc.fr_imidazole(mol)
            mol_graph.fr_ketone_Topliss = dsc.fr_ketone_Topliss(mol)
            mol_graph.PEOE_VSA7 = dsc.PEOE_VSA7(mol)
            mol_graph.fr_alkyl_halide = dsc.fr_alkyl_halide(mol)
            mol_graph.NumSaturatedHeterocycles = dsc.NumSaturatedHeterocycles(mol)
            # 31
            mol_graph.fr_methoxy = dsc.fr_methoxy(mol)
            mol_graph.fr_phos_acid = dsc.fr_phos_acid(mol)
            mol_graph.fr_pyridine = dsc.fr_pyridine(mol)
            mol_graph.MinAbsEStateIndex = dsc.MinAbsEStateIndex(mol)
            mol_graph.fr_para_hydroxylation = dsc.fr_para_hydroxylation(mol)
            # 36
            mol_graph.fr_phos_ester = dsc.fr_phos_ester(mol)
            mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            mol_graph.PEOE_VSA8 = dsc.PEOE_VSA8(mol)
            mol_graph.fr_Ndealkylation2 = dsc.fr_Ndealkylation2(mol)
            mol_graph.PEOE_VSA5 = dsc.PEOE_VSA5(mol)
            # 41
            mol_graph.fr_aryl_methyl = dsc.fr_aryl_methyl(mol)
            mol_graph.NumHDonors = dsc.NumHDonors(mol)
            mol_graph.fr_imide = dsc.fr_imide(mol)
            mol_graph.fr_priamide = dsc.fr_priamide(mol)
            mol_graph.RingCount = dsc.RingCount(mol)
            # 46
            mol_graph.SlogP_VSA8 = dsc.SlogP_VSA8(mol)
            mol_graph.VSA_EState4 = dsc.VSA_EState4(mol)
            mol_graph.SMR_VSA5 = dsc.SMR_VSA5(mol)
            mol_graph.FpDensityMorgan3 = dsc.FpDensityMorgan3(mol)
            mol_graph.FractionCSP3 = dsc.FractionCSP3(mol)

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    for feat in ['NHOHCount', 'SlogP_VSA2', 'SlogP_VSA10', 'NumAromaticRings', 'MaxEStateIndex', 
                'PEOE_VSA14', 'fr_Ar_NH', 'SMR_VSA3', 'SMR_VSA7', 'SlogP_VSA5', 
                'VSA_EState8', 'MaxAbsEStateIndex', 'PEOE_VSA2', 'fr_Nhpyrrole', 'fr_amide', 
                'SlogP_VSA3', 'BCUT2D_MRHI', 'fr_nitrile', 'MolLogP', 'PEOE_VSA10', 
                'MinPartialCharge', 'fr_Al_OH', 'fr_sulfone', 'fr_Al_COO', 'fr_nitro_arom_nonortho', 
                'fr_imidazole', 'fr_ketone_Topliss', 'PEOE_VSA7', 'fr_alkyl_halide', 'NumSaturatedHeterocycles', 
                'fr_methoxy', 'fr_phos_acid', 'fr_pyridine', 'MinAbsEStateIndex', 'fr_para_hydroxylation', 
                'fr_phos_ester', 'NumAromaticHeterocycles', 'PEOE_VSA8', 'fr_Ndealkylation2', 'PEOE_VSA5', 
                'fr_aryl_methyl', 'NumHDonors', 'fr_imide', 'fr_priamide', 'RingCount', 
                'SlogP_VSA8', 'VSA_EState4', 'SMR_VSA5', 'FpDensityMorgan3', 'FractionCSP3']:
        FeatureNormalization(mol_graphs, feat)

    return samples


# esol
def read_dataset_esol(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)

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

    return samples

# lipo
def read_dataset_lipo(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)

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

    for feat in ['MolLogP', 'fr_COO', 'Ipc', 'fr_sulfonamd', 'PEOE_VSA7',
                'PEOE_VSA13', 'SlogP_VSA10', 'fr_unbrch_alkane', 'SMR_VSA10', 'PEOE_VSA12',
                'fr_guanido', 'FpDensityMorgan1', 'NHOHCount', 'fr_sulfide', 'VSA_EState5',
                'fr_HOCCN', 'fr_piperdine', 'NumSaturatedCarbocycles', 'fr_amidine', 'NumHDonors',
                'NumAromaticRings', 'BalabanJ', 'NumAromaticHeterocycles', 'MinEStateIndex', 'fr_Ar_N']:
        FeatureNormalization(mol_graphs, feat)

    return samples


# Self-Curated Gas
def read_dataset_scgas(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=float)

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

    for feat in ['MolMR', 'TPSA', 'fr_halogen', 'SlogP_VSA12', 'RingCount', 
                'Kappa1', 'NumHAcceptors', 'NumHDonors', 'SMR_VSA7', 'SMR_VSA5',
                'Chi1', 'Chi3n', 'BertzCT', 'VSA_EState8', 'NumAliphaticCarbocycles', 
                'HallKierAlpha', 'VSA_EState6', 'NumAromaticRings', 'Chi4n', 'PEOE_VSA7', 
                'SlogP_VSA5', 'VSA_EState7', 'NOCount']:
        FeatureNormalization(mol_graphs, feat)

    return samples