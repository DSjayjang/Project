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
        targets = np.array(data_mat[:, 1], dtype = float)

        # Convert each SMILES to DGL grgaph
        for smile, tgt in zip(smiles, targets):
            graph_obj = MolGraph(smile)
            if graph_obj.mol is None:
                print('graph_obj.mol is None!')
                continue

            # 1
            graph_obj.MolLogP = dsc.MolLogP(graph_obj.mol)
            graph_obj.SMR_VSA10 = dsc.SMR_VSA10(graph_obj.mol)
            graph_obj.MaxEStateIndex = dsc.MaxEStateIndex(graph_obj.mol)
            graph_obj.MaxAbsPartialCharge = dsc.MaxAbsPartialCharge(graph_obj.mol)
            graph_obj.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(graph_obj.mol)
            # 6
            graph_obj.BCUT2D_MWLOW = dsc.BCUT2D_MWLOW(graph_obj.mol)
            graph_obj.fr_imide = dsc.fr_imide(graph_obj.mol)
            graph_obj.Kappa2 = dsc.Kappa2(graph_obj.mol)
            graph_obj.MinAbsPartialCharge = dsc.MinAbsPartialCharge(graph_obj.mol)
            graph_obj.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(graph_obj.mol)
            # 11
            graph_obj.SlogP_VSA1 = dsc.SlogP_VSA1(graph_obj.mol)
            graph_obj.fr_amide = dsc.fr_amide(graph_obj.mol)
            graph_obj.BalabanJ = dsc.BalabanJ(graph_obj.mol)
            graph_obj.fr_Ar_NH = dsc.fr_Ar_NH(graph_obj.mol)
            graph_obj.PEOE_VSA8 = dsc.PEOE_VSA8(graph_obj.mol)
            # 16
            graph_obj.NumSaturatedRings = dsc.NumSaturatedRings(graph_obj.mol)
            graph_obj.fr_NH0 = dsc.fr_NH0(graph_obj.mol)
            graph_obj.PEOE_VSA13 = dsc.PEOE_VSA13(graph_obj.mol)
            graph_obj.fr_barbitur = dsc.fr_barbitur(graph_obj.mol)
            graph_obj.fr_alkyl_halide = dsc.fr_alkyl_halide(graph_obj.mol)
            # 21
            graph_obj.fr_C_O = dsc.fr_C_O(graph_obj.mol)
            graph_obj.fr_bicyclic = dsc.fr_bicyclic(graph_obj.mol)
            graph_obj.fr_ester = dsc.fr_ester(graph_obj.mol)
            graph_obj.PEOE_VSA9 = dsc.PEOE_VSA9(graph_obj.mol)
            graph_obj.fr_Al_OH_noTert = dsc.fr_Al_OH_noTert(graph_obj.mol)
            # 26
            graph_obj.SlogP_VSA10 = dsc.SlogP_VSA10(graph_obj.mol)
            graph_obj.EState_VSA11 = dsc.EState_VSA11(graph_obj.mol)
            graph_obj.fr_imidazole = dsc.fr_imidazole(graph_obj.mol)
            graph_obj.EState_VSA10 = dsc.EState_VSA10(graph_obj.mol)
            graph_obj.EState_VSA5 = dsc.EState_VSA5(graph_obj.mol)
            # 31
            graph_obj.SMR_VSA9 = dsc.SMR_VSA9(graph_obj.mol)
            graph_obj.FractionCSP3 = dsc.FractionCSP3(graph_obj.mol)
            graph_obj.FpDensityMorgan2 = dsc.FpDensityMorgan2(graph_obj.mol)
            graph_obj.fr_furan = dsc.fr_furan(graph_obj.mol)
            graph_obj.fr_hdrzine = dsc.fr_hdrzine(graph_obj.mol)
            # 36
            graph_obj.fr_aryl_methyl = dsc.fr_aryl_methyl(graph_obj.mol)
            graph_obj.EState_VSA8 = dsc.EState_VSA8(graph_obj.mol)
            graph_obj.fr_phos_acid = dsc.fr_phos_acid(graph_obj.mol)
            graph_obj.SlogP_VSA7 = dsc.SlogP_VSA7(graph_obj.mol)
            graph_obj.SlogP_VSA4 = dsc.SlogP_VSA4(graph_obj.mol)
            # 41
            graph_obj.EState_VSA2 = dsc.EState_VSA2(graph_obj.mol)
            graph_obj.fr_nitro_arom_nonortho = dsc.fr_nitro_arom_nonortho(graph_obj.mol)
            graph_obj.fr_para_hydroxylation = dsc.fr_para_hydroxylation(graph_obj.mol)
            
            self.data.append((graph_obj, tgt))
            self.graphs.append(graph_obj)

        for feat in ['MolLogP','SMR_VSA10','MaxEStateIndex','MaxAbsPartialCharge','BCUT2D_CHGHI','BCUT2D_MWLOW','fr_imide','Kappa2','MinAbsPartialCharge','NumAromaticHeterocycles','SlogP_VSA1','fr_amide','BalabanJ','fr_Ar_NH','PEOE_VSA8','NumSaturatedRings','fr_NH0','PEOE_VSA13','fr_barbitur','fr_alkyl_halide','fr_C_O','fr_bicyclic','fr_ester','PEOE_VSA9','fr_Al_OH_noTert','SlogP_VSA10','EState_VSA11','fr_imidazole','EState_VSA10','EState_VSA5','SMR_VSA9','FractionCSP3','FpDensityMorgan2','fr_furan','fr_hdrzine','fr_aryl_methyl','EState_VSA8','fr_phos_acid','SlogP_VSA7','SlogP_VSA4','EState_VSA2','fr_nitro_arom_nonortho','fr_para_hydroxylation']:
            FeatureNormalization(self.graphs, feat)

    def __len__(self):
        return (len(self.data))
    