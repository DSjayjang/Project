import numpy as np
import pandas as pd
import torch
import rdkit.Chem.Descriptors as dsc

from utils.mol_props import load_atomic_props
from utils.utils import FeatureNormalization
from utils.mol_graph import smiles_to_mol_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이것을 이제 DATASET을 입력받는 클래스로 만들어야 함
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

def read_dataset_esol(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]
#    target = np.array(data_mat[:, 1:3], dtype=np.float)
    target = np.array(data_mat[:, 1:3], dtype=float)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            ####################################################
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

            ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    ####################################################
    # 1
    FeatureNormalization(mol_graphs, 'MolLogP')
    FeatureNormalization(mol_graphs, 'MaxAbsPartialCharge')
    FeatureNormalization(mol_graphs, 'MaxEStateIndex')
    FeatureNormalization(mol_graphs, 'SMR_VSA10')
    FeatureNormalization(mol_graphs, 'Kappa2')
    # 6
    FeatureNormalization(mol_graphs, 'BCUT2D_MWLOW')
    FeatureNormalization(mol_graphs, 'PEOE_VSA13')
    FeatureNormalization(mol_graphs, 'MinAbsPartialCharge')
    FeatureNormalization(mol_graphs, 'BCUT2D_CHGHI')
    FeatureNormalization(mol_graphs, 'PEOE_VSA6')
    # 11
    FeatureNormalization(mol_graphs, 'SlogP_VSA1')
    FeatureNormalization(mol_graphs, 'fr_nitro')
    FeatureNormalization(mol_graphs, 'BalabanJ')
    FeatureNormalization(mol_graphs, 'SMR_VSA9')
    FeatureNormalization(mol_graphs, 'fr_alkyl_halide')
    # 16
    FeatureNormalization(mol_graphs, 'fr_hdrzine')
    FeatureNormalization(mol_graphs, 'PEOE_VSA8')
    FeatureNormalization(mol_graphs, 'fr_Ar_NH')
    FeatureNormalization(mol_graphs, 'fr_imidazole')
    FeatureNormalization(mol_graphs, 'fr_Nhpyrrole')
    # 21
    FeatureNormalization(mol_graphs, 'EState_VSA5')
    FeatureNormalization(mol_graphs, 'PEOE_VSA4')
    FeatureNormalization(mol_graphs, 'fr_ester')
    FeatureNormalization(mol_graphs, 'PEOE_VSA2')
    FeatureNormalization(mol_graphs, 'NumAromaticCarbocycles')
    # 26
    FeatureNormalization(mol_graphs, 'BCUT2D_LOGPHI')
    FeatureNormalization(mol_graphs, 'EState_VSA11')
    FeatureNormalization(mol_graphs, 'fr_furan')
    FeatureNormalization(mol_graphs, 'EState_VSA2')
    FeatureNormalization(mol_graphs, 'fr_benzene')
    # 31
    FeatureNormalization(mol_graphs, 'fr_sulfide')
    FeatureNormalization(mol_graphs, 'fr_aryl_methyl')
    FeatureNormalization(mol_graphs, 'SlogP_VSA10')
    FeatureNormalization(mol_graphs, 'HeavyAtomMolWt')
    FeatureNormalization(mol_graphs, 'fr_nitro_arom_nonortho')
    # 36
    FeatureNormalization(mol_graphs, 'FpDensityMorgan2')
    FeatureNormalization(mol_graphs, 'EState_VSA8')
    FeatureNormalization(mol_graphs, 'fr_bicyclic')
    FeatureNormalization(mol_graphs, 'fr_aniline')
    FeatureNormalization(mol_graphs, 'fr_allylic_oxid')
    # 41
    FeatureNormalization(mol_graphs, 'fr_C_S')
    FeatureNormalization(mol_graphs, 'SlogP_VSA7')
    FeatureNormalization(mol_graphs, 'SlogP_VSA4')
    FeatureNormalization(mol_graphs, 'fr_para_hydroxylation')
    FeatureNormalization(mol_graphs, 'PEOE_VSA7')
    # 46
    FeatureNormalization(mol_graphs, 'fr_Al_OH_noTert')
    FeatureNormalization(mol_graphs, 'fr_pyridine')
    FeatureNormalization(mol_graphs, 'fr_phos_acid')
    FeatureNormalization(mol_graphs, 'fr_phos_ester')
    FeatureNormalization(mol_graphs, 'NumAromaticHeterocycles')
    # 51
    FeatureNormalization(mol_graphs, 'EState_VSA7')
    FeatureNormalization(mol_graphs, 'PEOE_VSA12')
    FeatureNormalization(mol_graphs, 'Ipc')
    FeatureNormalization(mol_graphs, 'FpDensityMorgan1')
    FeatureNormalization(mol_graphs, 'PEOE_VSA14')
    # 56
    FeatureNormalization(mol_graphs, 'fr_guanido')
    FeatureNormalization(mol_graphs, 'fr_benzodiazepine')
    FeatureNormalization(mol_graphs, 'fr_thiophene')
    FeatureNormalization(mol_graphs, 'fr_Ndealkylation1')
    FeatureNormalization(mol_graphs, 'fr_aldehyde')
    # 61
    FeatureNormalization(mol_graphs, 'fr_term_acetylene')
    FeatureNormalization(mol_graphs, 'SMR_VSA2')
    FeatureNormalization(mol_graphs, 'fr_lactone')
    
    return samples


##############################################################################################################################
##################### 여긴 테스트중이고 #####################

from torch.utils.data import Dataset

class MoleculeDataset(Dataset):
    def __init__(self, file_name):
        self.samples = []
        self.mol_graphs = []

        file_name += '.csv'
        # CSV 불러오기
        data_mat = np.array(pd.read_csv(file_name))
        smiles = data_mat[:, 0]
        target = np.array(data_mat[:, 1:3], dtype=float)

        # SMILES → mol, graph 변환
        for i in range(data_mat.shape[0]):
            mol, mol_graph = smiles_to_mol_graph(smiles[i])

            if mol is not None and mol_graph is not None:
                mol_graph.num_atoms = mol.GetNumAtoms()
                mol_graph.weight = dsc.ExactMolWt(mol)
                mol_graph.num_rings = mol.GetRingInfo().NumRings()

                self.samples.append((mol_graph, target[i]))
                self.mol_graphs.append(mol_graph)

        # Feature 정규화
        for feat in ['num_atoms', 'weight', 'num_rings']:
            FeatureNormalization(self.mol_graphs, feat)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]