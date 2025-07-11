import pandas
import torch
import dgl
import numpy as np
import rdkit.Chem.Descriptors as dsc
from rdkit import Chem
from utils import utils
#from mendeleev import get_table
from mendeleev.fetch import fetch_table
import traceback
from utils.utils import FeatureNormalization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sel_prop_names = ['atomic_weight',
                'atomic_radius',
                'atomic_volume',
                'dipole_polarizability',
                'fusion_heat',
                'thermal_conductivity',
                'vdw_radius',
                'en_pauling']
dim_atomic_feat = len(sel_prop_names)
dim_self_feat = 43


class molDGLGraph(dgl.DGLGraph):
    def __init__(self, smiles, adj_mat, feat_mat, mol):
        super(molDGLGraph, self).__init__()
        self.smiles = smiles
        self.adj_mat = adj_mat
        self.feat_mat = feat_mat
        self.atomic_nodes = []
        self.neighbors = {}

        node_id = 0
        for atom in mol.GetAtoms():
            self.atomic_nodes.append(atom.GetSymbol())
            self.neighbors[node_id] = atoms_to_symbols(atom.GetNeighbors())
            node_id += 1


def read_atom_prop():
#    tb_atomic_props = get_table('elements')
    tb_atomic_props = fetch_table('elements')
#    arr_atomic_nums = np.array(tb_atomic_props['atomic_number'], dtype=np.int)
    arr_atomic_nums = np.array(tb_atomic_props['atomic_number'], dtype=int)
#    arr_atomic_props = np.nan_to_num(np.array(tb_atomic_props[sel_prop_names], dtype=np.float32))
    arr_atomic_props = np.nan_to_num(np.array(tb_atomic_props[sel_prop_names], dtype=float))
    arr_atomic_props = utils.Z_Score(arr_atomic_props)
    atomic_props_mat = {arr_atomic_nums[i]: arr_atomic_props[i, :] for i in range(0, arr_atomic_nums.shape[0])}

    return atomic_props_mat


def construct_mol_graph(smiles, mol, adj_mat, feat_mat):
    molGraph = molDGLGraph(smiles, adj_mat, feat_mat, mol).to(device)
    edges = utils.adj_mat_to_edges(adj_mat)
    src, dst = tuple(zip(*edges))

    molGraph.add_nodes(adj_mat.shape[0])
    molGraph.add_edges(src, dst)
    molGraph.ndata['feat'] = torch.tensor(feat_mat, dtype=torch.float32).to(device)

    return molGraph


def smiles_to_mol_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        node_feat_mat = np.empty([mol.GetNumAtoms(), atomic_props.get(1).shape[0]])

        ind = 0
        for atom in mol.GetAtoms():
            node_feat_mat[ind, :] = atomic_props.get(atom.GetAtomicNum())
            ind = ind + 1

        return mol, construct_mol_graph(smiles, mol, adj_mat, node_feat_mat)
    except Exception as e:
        print(f"Error processing SMILES: {smiles}")
        print(traceback.format_exc())  # 예외 정보를 출력
        return None, None


def atoms_to_symbols(atoms):
    symbols = []

    for atom in atoms:
        symbols.append(atom.GetSymbol())

    return symbols


def normalize_self_feat(mol_graphs, self_feat_name):
    self_feats = []

    for mol_graph in mol_graphs:
        self_feats.append(getattr(mol_graph, self_feat_name))

    mean_self_feat = np.mean(self_feats)
    std_self_feat = np.std(self_feats)

    for mol_graph in mol_graphs:
        if std_self_feat == 0:
            setattr(mol_graph, self_feat_name, 0)
        else:
            setattr(mol_graph, self_feat_name, (getattr(mol_graph, self_feat_name) - mean_self_feat) / std_self_feat)

def read_dataset(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pandas.read_csv(file_name))
    smiles = data_mat[:, 0]

    target = np.array(data_mat[:, 1], dtype=float)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

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

    ####################################################
    # for feat in ['MolLogP','SMR_VSA10','MaxEStateIndex','MaxAbsPartialCharge','BCUT2D_CHGHI','BCUT2D_MWLOW','fr_imide','Kappa2','MinAbsPartialCharge','NumAromaticHeterocycles','SlogP_VSA1','fr_amide','BalabanJ','fr_Ar_NH','PEOE_VSA8','NumSaturatedRings','fr_NH0','PEOE_VSA13','fr_barbitur','fr_alkyl_halide','fr_C_O','fr_bicyclic','fr_ester','PEOE_VSA9','fr_Al_OH_noTert','SlogP_VSA10','EState_VSA11','fr_imidazole','EState_VSA10','EState_VSA5','SMR_VSA9','FractionCSP3','FpDensityMorgan2','fr_furan','fr_hdrzine','fr_aryl_methyl','EState_VSA8','fr_phos_acid','SlogP_VSA7','SlogP_VSA4','EState_VSA2','fr_nitro_arom_nonortho','fr_para_hydroxylation']:
    #     FeatureNormalization(mol_graphs, feat)

    ####################################################
    # 1
    normalize_self_feat(mol_graphs, 'MolLogP')
    normalize_self_feat(mol_graphs, 'SMR_VSA10')
    normalize_self_feat(mol_graphs, 'MaxEStateIndex')
    normalize_self_feat(mol_graphs, 'MaxAbsPartialCharge')
    normalize_self_feat(mol_graphs, 'BCUT2D_CHGHI')
    # 6
    normalize_self_feat(mol_graphs, 'BCUT2D_MWLOW')
    normalize_self_feat(mol_graphs, 'fr_imide')
    normalize_self_feat(mol_graphs, 'Kappa2')
    normalize_self_feat(mol_graphs, 'MinAbsPartialCharge')
    normalize_self_feat(mol_graphs, 'NumAromaticHeterocycles')
    # 11
    normalize_self_feat(mol_graphs, 'SlogP_VSA1')
    normalize_self_feat(mol_graphs, 'fr_amide')
    normalize_self_feat(mol_graphs, 'BalabanJ')
    normalize_self_feat(mol_graphs, 'fr_Ar_NH')
    normalize_self_feat(mol_graphs, 'PEOE_VSA8')
    # 16
    normalize_self_feat(mol_graphs, 'NumSaturatedRings')
    normalize_self_feat(mol_graphs, 'fr_NH0')
    normalize_self_feat(mol_graphs, 'PEOE_VSA13')
    normalize_self_feat(mol_graphs, 'fr_barbitur')
    normalize_self_feat(mol_graphs, 'fr_alkyl_halide')
    # 21
    normalize_self_feat(mol_graphs, 'fr_C_O')
    normalize_self_feat(mol_graphs, 'fr_bicyclic')
    normalize_self_feat(mol_graphs, 'fr_ester')
    normalize_self_feat(mol_graphs, 'PEOE_VSA9')
    normalize_self_feat(mol_graphs, 'fr_Al_OH_noTert')
    # 26
    normalize_self_feat(mol_graphs, 'SlogP_VSA10')
    normalize_self_feat(mol_graphs, 'EState_VSA11')
    normalize_self_feat(mol_graphs, 'fr_imidazole')
    normalize_self_feat(mol_graphs, 'EState_VSA10')
    normalize_self_feat(mol_graphs, 'EState_VSA5')
    # 31
    normalize_self_feat(mol_graphs, 'SMR_VSA9')
    normalize_self_feat(mol_graphs, 'FractionCSP3')
    normalize_self_feat(mol_graphs, 'FpDensityMorgan2')
    normalize_self_feat(mol_graphs, 'fr_furan')
    normalize_self_feat(mol_graphs, 'fr_hdrzine')
    # 36
    normalize_self_feat(mol_graphs, 'fr_aryl_methyl')
    normalize_self_feat(mol_graphs, 'EState_VSA8')
    normalize_self_feat(mol_graphs, 'fr_phos_acid')
    normalize_self_feat(mol_graphs, 'SlogP_VSA7')
    normalize_self_feat(mol_graphs, 'SlogP_VSA4')
    # 41
    normalize_self_feat(mol_graphs, 'EState_VSA2')
    normalize_self_feat(mol_graphs, 'fr_nitro_arom_nonortho')
    normalize_self_feat(mol_graphs, 'fr_para_hydroxylation')

    return samples

atomic_props = read_atom_prop()