import os
import pandas as pd
import numpy as np
import rdkit.Chem.Descriptors as dsc

from utils.utils import FeatureNormalization
from utils.mol_graph import MolGraph

# freesolv
class MoleculeDataset_freesolv:
    """
    내용 확인
    Reads a CSV file of SMILES and targets, constructs DGL graphs with atomic and molecular features,
    and applies z-score normalization to self-features.
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []
        self.graphs = []
        self._load_and_process()
    
    def _load_and_process(self):
        # Data Load
        data_mat = np.array(pd.read_csv(self.file_name + '.csv'))
        smiles = data_mat[:, 0]
        targets = np.array(data_mat[:, 1], dtype = float)

        # Convert each SMILES to DGL grgaph
        for smile, tgt in zip(smiles, targets):
            graph_obj = MolGraph(smile)
            if graph_obj.mol is None:
                print('graph_obj.mol is None!')
                continue

            # 1
            graph_obj.NHOHCount = dsc.NHOHCount(graph_obj.mol)
            graph_obj.SMR_VSA5 = dsc.SMR_VSA5(graph_obj.mol)
            graph_obj.SlogP_VSA2 = dsc.SlogP_VSA2(graph_obj.mol)
            graph_obj.TPSA = dsc.TPSA(graph_obj.mol)
            graph_obj.MaxEStateIndex = dsc.MaxEStateIndex(graph_obj.mol)
            # 6
            graph_obj.fr_Ar_NH = dsc.fr_Ar_NH(graph_obj.mol)
            graph_obj.Chi2v = dsc.Chi2v(graph_obj.mol)
            graph_obj.SlogP_VSA10 = dsc.SlogP_VSA10(graph_obj.mol)
            graph_obj.NumHeteroatoms = dsc.NumHeteroatoms(graph_obj.mol)
            graph_obj.RingCount = dsc.RingCount(graph_obj.mol)
            # 11
            graph_obj.fr_amide = dsc.fr_amide(graph_obj.mol)
            graph_obj.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(graph_obj.mol)
            graph_obj.PEOE_VSA14 = dsc.PEOE_VSA14(graph_obj.mol)
            graph_obj.SlogP_VSA4 = dsc.SlogP_VSA4(graph_obj.mol)
            graph_obj.VSA_EState8 = dsc.VSA_EState8(graph_obj.mol)
            # 16
            graph_obj.PEOE_VSA2 = dsc.PEOE_VSA2(graph_obj.mol)
            graph_obj.PEOE_VSA10 = dsc.PEOE_VSA10(graph_obj.mol)
            graph_obj.fr_Al_OH = dsc.fr_Al_OH(graph_obj.mol)
            graph_obj.fr_bicyclic = dsc.fr_bicyclic(graph_obj.mol)
            graph_obj.SMR_VSA2 = dsc.SMR_VSA2(graph_obj.mol)
            # 21
            graph_obj.PEOE_VSA7 = dsc.PEOE_VSA7(graph_obj.mol)
            graph_obj.MinPartialCharge = dsc.MinPartialCharge(graph_obj.mol)
            graph_obj.fr_aryl_methyl = dsc.fr_aryl_methyl(graph_obj.mol)
            graph_obj.NumSaturatedHeterocycles = dsc.NumSaturatedHeterocycles(graph_obj.mol)
            graph_obj.NumHDonors = dsc.NumHDonors(graph_obj.mol)
            # 26
            graph_obj.fr_imidazole = dsc.fr_imidazole(graph_obj.mol)
            graph_obj.fr_phos_ester = dsc.fr_phos_ester(graph_obj.mol)
            graph_obj.fr_Al_COO = dsc.fr_Al_COO(graph_obj.mol)
            graph_obj.EState_VSA6 = dsc.EState_VSA6(graph_obj.mol)
            graph_obj.PEOE_VSA8 = dsc.PEOE_VSA8(graph_obj.mol)
            # 31
            graph_obj.fr_ketone_Topliss = dsc.fr_ketone_Topliss(graph_obj.mol)
            graph_obj.fr_imide = dsc.fr_imide(graph_obj.mol)
            graph_obj.fr_nitro_arom_nonortho = dsc.fr_nitro_arom_nonortho(graph_obj.mol)
            graph_obj.EState_VSA8 = dsc.EState_VSA8(graph_obj.mol)
            graph_obj.fr_para_hydroxylation = dsc.fr_para_hydroxylation(graph_obj.mol)
            # 36
            graph_obj.Kappa2 = dsc.Kappa2(graph_obj.mol)
            graph_obj.Ipc = dsc.Ipc(graph_obj.mol)
            
            self.data.append((graph_obj, tgt))
            self.graphs.append(graph_obj)

        for feat in ['NHOHCount', 'SMR_VSA5', 'SlogP_VSA2', 'TPSA', 'MaxEStateIndex', 'fr_Ar_NH', 'Chi2v', 'SlogP_VSA10', 'NumHeteroatoms', 'RingCount', 'fr_amide',
                     'NumAromaticHeterocycles', 'PEOE_VSA14', 'SlogP_VSA4', 'VSA_EState8', 'PEOE_VSA2', 'PEOE_VSA10', 'fr_Al_OH', 'fr_bicyclic', 'SMR_VSA2',
                     'PEOE_VSA7', 'MinPartialCharge', 'fr_aryl_methyl', 'NumSaturatedHeterocycles', 'NumHDonors', 'fr_imidazole', 'fr_phos_ester', 'fr_Al_COO', 'EState_VSA6', 'PEOE_VSA8',
                     'fr_ketone_Topliss', 'fr_imide', 'fr_nitro_arom_nonortho', 'EState_VSA8', 'fr_para_hydroxylation', 'Kappa2', 'Ipc']:
            FeatureNormalization(self.graphs, feat)
    
    def __len__(self):
        return (len(self.data))
    
# esol
class MoleculeDataset_esol:
    """
    내용 확인
    Reads a CSV file of SMILES and targets, constructs DGL graphs with atomic and molecular features,
    and applies z-score normalization to self-features.
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []
        self.graphs = []
        self._load_and_process()
    
    def _load_and_process(self):
        # Data Load
        data_mat = np.array(pd.read_csv(self.file_name + '.csv'))
        smiles = data_mat[:, 0]
        targets = np.array(data_mat[:, 1:3], dtype = float)

        # Convert each SMILES to DGL grgaph
        for smile, tgt in zip(smiles, targets):
            graph_obj = MolGraph(smile)
            if graph_obj.mol is None:
                print('graph_obj.mol is None!')
                continue

            # 1
            graph_obj.MolLogP = dsc.MolLogP(graph_obj.mol)
            graph_obj.MaxAbsPartialCharge = dsc.MaxAbsPartialCharge(graph_obj.mol)
            graph_obj.MaxEStateIndex = dsc.MaxEStateIndex(graph_obj.mol)
            graph_obj.SMR_VSA10 = dsc.SMR_VSA10(graph_obj.mol)
            graph_obj.Kappa2 = dsc.Kappa2(graph_obj.mol)
            # 6
            graph_obj.BCUT2D_MWLOW = dsc.BCUT2D_MWLOW(graph_obj.mol)
            graph_obj.PEOE_VSA13 = dsc.PEOE_VSA13(graph_obj.mol)
            graph_obj.MinAbsPartialCharge = dsc.MinAbsPartialCharge(graph_obj.mol)
            graph_obj.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(graph_obj.mol)
            graph_obj.PEOE_VSA6 = dsc.PEOE_VSA6(graph_obj.mol)
            # 11
            graph_obj.SlogP_VSA1 = dsc.SlogP_VSA1(graph_obj.mol)
            graph_obj.fr_nitro = dsc.fr_nitro(graph_obj.mol)
            graph_obj.BalabanJ = dsc.BalabanJ(graph_obj.mol)
            graph_obj.SMR_VSA9 = dsc.SMR_VSA9(graph_obj.mol)
            graph_obj.fr_alkyl_halide = dsc.fr_alkyl_halide(graph_obj.mol)
            # 16
            graph_obj.fr_hdrzine = dsc.fr_hdrzine(graph_obj.mol)
            graph_obj.PEOE_VSA8 = dsc.PEOE_VSA8(graph_obj.mol)
            graph_obj.fr_Ar_NH = dsc.fr_Ar_NH(graph_obj.mol)
            graph_obj.fr_imidazole = dsc.fr_imidazole(graph_obj.mol)
            graph_obj.fr_Nhpyrrole = dsc.fr_Nhpyrrole(graph_obj.mol)
            # 21
            graph_obj.EState_VSA5 = dsc.EState_VSA5(graph_obj.mol)
            graph_obj.PEOE_VSA4 = dsc.PEOE_VSA4(graph_obj.mol)
            graph_obj.fr_ester = dsc.fr_ester(graph_obj.mol)
            graph_obj.PEOE_VSA2 = dsc.PEOE_VSA2(graph_obj.mol)
            graph_obj.NumAromaticCarbocycles = dsc.NumAromaticCarbocycles(graph_obj.mol)
            # 26
            graph_obj.BCUT2D_LOGPHI = dsc.BCUT2D_LOGPHI(graph_obj.mol)
            graph_obj.EState_VSA11 = dsc.EState_VSA11(graph_obj.mol)
            graph_obj.fr_furan = dsc.fr_furan(graph_obj.mol)
            graph_obj.EState_VSA2 = dsc.EState_VSA2(graph_obj.mol)
            graph_obj.fr_benzene = dsc.fr_benzene(graph_obj.mol)
            # 31
            graph_obj.fr_sulfide = dsc.fr_sulfide(graph_obj.mol)
            graph_obj.fr_aryl_methyl = dsc.fr_aryl_methyl(graph_obj.mol)
            graph_obj.SlogP_VSA10 = dsc.SlogP_VSA10(graph_obj.mol)
            graph_obj.HeavyAtomMolWt = dsc.HeavyAtomMolWt(graph_obj.mol)
            graph_obj.fr_nitro_arom_nonortho = dsc.fr_nitro_arom_nonortho(graph_obj.mol)
            # 36
            graph_obj.FpDensityMorgan2 = dsc.FpDensityMorgan2(graph_obj.mol)
            graph_obj.EState_VSA8 = dsc.EState_VSA8(graph_obj.mol)
            graph_obj.fr_bicyclic = dsc.fr_bicyclic(graph_obj.mol)
            graph_obj.fr_aniline = dsc.fr_aniline(graph_obj.mol)
            graph_obj.fr_allylic_oxid = dsc.fr_allylic_oxid(graph_obj.mol)
            # 41
            graph_obj.fr_C_S = dsc.fr_C_S(graph_obj.mol)
            graph_obj.SlogP_VSA7 = dsc.SlogP_VSA7(graph_obj.mol)
            graph_obj.SlogP_VSA4 = dsc.SlogP_VSA4(graph_obj.mol)
            graph_obj.fr_para_hydroxylation = dsc.fr_para_hydroxylation(graph_obj.mol)
            graph_obj.PEOE_VSA7 = dsc.PEOE_VSA7(graph_obj.mol)
            # 46
            graph_obj.fr_Al_OH_noTert = dsc.fr_Al_OH_noTert(graph_obj.mol)
            graph_obj.fr_pyridine = dsc.fr_pyridine(graph_obj.mol)
            graph_obj.fr_phos_acid = dsc.fr_phos_acid(graph_obj.mol)
            graph_obj.fr_phos_ester = dsc.fr_phos_ester(graph_obj.mol)
            graph_obj.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(graph_obj.mol)
            # 51
            graph_obj.EState_VSA7 = dsc.EState_VSA7(graph_obj.mol)
            graph_obj.PEOE_VSA12 = dsc.PEOE_VSA12(graph_obj.mol)
            graph_obj.Ipc = dsc.Ipc(graph_obj.mol)
            graph_obj.FpDensityMorgan1 = dsc.FpDensityMorgan1(graph_obj.mol)
            graph_obj.PEOE_VSA14 = dsc.PEOE_VSA14(graph_obj.mol)
            # 56
            graph_obj.fr_guanido = dsc.fr_guanido(graph_obj.mol)
            graph_obj.fr_benzodiazepine = dsc.fr_benzodiazepine(graph_obj.mol)
            graph_obj.fr_thiophene = dsc.fr_thiophene(graph_obj.mol)
            graph_obj.fr_Ndealkylation1 = dsc.fr_Ndealkylation1(graph_obj.mol)
            graph_obj.fr_aldehyde = dsc.fr_aldehyde(graph_obj.mol)
            # 61
            graph_obj.fr_term_acetylene = dsc.fr_term_acetylene(graph_obj.mol)
            graph_obj.SMR_VSA2 = dsc.SMR_VSA2(graph_obj.mol)
            graph_obj.fr_lactone = dsc.fr_lactone(graph_obj.mol)
            
            self.data.append((graph_obj, tgt))
            self.graphs.append(graph_obj)

        for feat in ['MolLogP', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'SMR_VSA10',
       'Kappa2', 'BCUT2D_MWLOW', 'PEOE_VSA13', 'MinAbsPartialCharge',
       'BCUT2D_CHGHI', 'PEOE_VSA6', 'SlogP_VSA1', 'fr_nitro', 'BalabanJ',
       'SMR_VSA9', 'fr_alkyl_halide', 'fr_hdrzine', 'PEOE_VSA8', 'fr_Ar_NH',
       'fr_imidazole', 'fr_Nhpyrrole', 'EState_VSA5', 'PEOE_VSA4', 'fr_ester',
       'PEOE_VSA2', 'NumAromaticCarbocycles', 'BCUT2D_LOGPHI', 'EState_VSA11',
       'fr_furan', 'EState_VSA2', 'fr_benzene', 'fr_sulfide', 'fr_aryl_methyl',
       'SlogP_VSA10', 'HeavyAtomMolWt', 'fr_nitro_arom_nonortho',
       'FpDensityMorgan2', 'EState_VSA8', 'fr_bicyclic', 'fr_aniline',
       'fr_allylic_oxid', 'fr_C_S', 'SlogP_VSA7', 'SlogP_VSA4',
       'fr_para_hydroxylation', 'PEOE_VSA7', 'fr_Al_OH_noTert', 'fr_pyridine',
       'fr_phos_acid', 'fr_phos_ester', 'NumAromaticHeterocycles',
       'EState_VSA7', 'PEOE_VSA12', 'Ipc', 'FpDensityMorgan1', 'PEOE_VSA14',
       'fr_guanido', 'fr_benzodiazepine', 'fr_thiophene', 'fr_Ndealkylation1',
       'fr_aldehyde', 'fr_term_acetylene', 'SMR_VSA2', 'fr_lactone']:
            FeatureNormalization(self.graphs, feat)

    def __len__(self):
        return (len(self.data))
    