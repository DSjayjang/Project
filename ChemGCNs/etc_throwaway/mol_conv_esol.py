import pandas
import torch
import dgl
import numpy as np
import rdkit.Chem.Descriptors as dsc
from rdkit import Chem
from utils import utils
from utils.utils import FeatureNormalization
from utils import mol_props
# from utils.mol_graph import MolGraph
import traceback

# from utils.molecule import MolGraph
from utils.utils import atoms_to_symbols
from utils.mol_graph import smiles_to_mol_graph
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dim_self_feat = 43

def read_dataset(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pandas.read_csv(file_name))
    smiles = data_mat[:, 0]
#    target = np.array(data_mat[:, 1:3], dtype=np.float)
    target = np.array(data_mat[:, 1:3], dtype=float)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])
        # mol, mol_graph = MolGraph.from_smiles(smiles[i])

        if mol is not None and mol_graph is not None:
            ####################################################
            # 1
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.SMR_VSA10 = dsc.SMR_VSA10(mol)
            mol_graph.MaxEStateIndex = dsc.MaxEStateIndex(mol)
            mol_graph.MaxAbsPartialCharge = dsc.MaxAbsPartialCharge(mol)
            mol_graph.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(mol)
            # 6
            mol_graph.BCUT2D_MWLOW = dsc.BCUT2D_MWLOW(mol)
            mol_graph.fr_imide = dsc.fr_imide(mol)
            mol_graph.Kappa2 = dsc.Kappa2(mol)
            mol_graph.MinAbsPartialCharge = dsc.MinAbsPartialCharge(mol)
            mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            # 11
            mol_graph.SlogP_VSA1 = dsc.SlogP_VSA1(mol)
            mol_graph.fr_amide = dsc.fr_amide(mol)
            mol_graph.BalabanJ = dsc.BalabanJ(mol)
            mol_graph.fr_Ar_NH = dsc.fr_Ar_NH(mol)
            mol_graph.PEOE_VSA8 = dsc.PEOE_VSA8(mol)
            # 16
            mol_graph.NumSaturatedRings = dsc.NumSaturatedRings(mol)
            mol_graph.fr_NH0 = dsc.fr_NH0(mol)
            mol_graph.PEOE_VSA13 = dsc.PEOE_VSA13(mol)
            mol_graph.fr_barbitur = dsc.fr_barbitur(mol)
            mol_graph.fr_alkyl_halide = dsc.fr_alkyl_halide(mol)
            # 21
            mol_graph.fr_C_O = dsc.fr_C_O(mol)
            mol_graph.fr_bicyclic = dsc.fr_bicyclic(mol)
            mol_graph.fr_ester = dsc.fr_ester(mol)
            mol_graph.PEOE_VSA9 = dsc.PEOE_VSA9(mol)
            mol_graph.fr_Al_OH_noTert = dsc.fr_Al_OH_noTert(mol)
            # 26
            mol_graph.SlogP_VSA10 = dsc.SlogP_VSA10(mol)
            mol_graph.EState_VSA11 = dsc.EState_VSA11(mol)
            mol_graph.fr_imidazole = dsc.fr_imidazole(mol)
            mol_graph.EState_VSA10 = dsc.EState_VSA10(mol)
            mol_graph.EState_VSA5 = dsc.EState_VSA5(mol)
            # 31
            mol_graph.SMR_VSA9 = dsc.SMR_VSA9(mol)
            mol_graph.FractionCSP3 = dsc.FractionCSP3(mol)
            mol_graph.FpDensityMorgan2 = dsc.FpDensityMorgan2(mol)
            mol_graph.fr_furan = dsc.fr_furan(mol)
            mol_graph.fr_hdrzine = dsc.fr_hdrzine(mol)
            # 36
            mol_graph.fr_aryl_methyl = dsc.fr_aryl_methyl(mol)
            mol_graph.EState_VSA8 = dsc.EState_VSA8(mol)
            mol_graph.fr_phos_acid = dsc.fr_phos_acid(mol)
            mol_graph.SlogP_VSA7 = dsc.SlogP_VSA7(mol)
            mol_graph.SlogP_VSA4 = dsc.SlogP_VSA4(mol)
            # 41
            mol_graph.EState_VSA2 = dsc.EState_VSA2(mol)
            mol_graph.fr_nitro_arom_nonortho = dsc.fr_nitro_arom_nonortho(mol)
            mol_graph.fr_para_hydroxylation = dsc.fr_para_hydroxylation(mol)
            ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)


    for feat in ['MolLogP','SMR_VSA10','MaxEStateIndex','MaxAbsPartialCharge','BCUT2D_CHGHI','BCUT2D_MWLOW','fr_imide','Kappa2','MinAbsPartialCharge','NumAromaticHeterocycles','SlogP_VSA1','fr_amide','BalabanJ','fr_Ar_NH','PEOE_VSA8','NumSaturatedRings','fr_NH0','PEOE_VSA13','fr_barbitur','fr_alkyl_halide','fr_C_O','fr_bicyclic','fr_ester','PEOE_VSA9','fr_Al_OH_noTert','SlogP_VSA10','EState_VSA11','fr_imidazole','EState_VSA10','EState_VSA5','SMR_VSA9','FractionCSP3','FpDensityMorgan2','fr_furan','fr_hdrzine','fr_aryl_methyl','EState_VSA8','fr_phos_acid','SlogP_VSA7','SlogP_VSA4','EState_VSA2','fr_nitro_arom_nonortho','fr_para_hydroxylation']:
        FeatureNormalization(mol_graphs, feat)

    return samples

atomic_props = mol_props.props