import numpy as np
import dgl
from rdkit import Chem
from utils.mol_props import props
from utils.utils import adj_mat_to_edges
import traceback
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def atoms_to_symbols(atoms):
    symbols = []

    for atom in atoms:
        symbols.append(atom.GetSymbol())

    return symbols

class MolGraph(dgl.DGLGraph):
    def __init__(self, smiles):
        super(MolGraph, self).__init__()
        self.smiles = smiles
        self.atomic_nodes = []
        self.neighbors = {}
        # try:
        self.mol = Chem.MolFromSmiles(smiles)
        
        self.adj_mat = Chem.GetAdjacencyMatrix(self.mol)
        self.feat_mat = np.empty([self.mol.GetNumAtoms(), props.get(1).shape[0]])
        ind = 0
        for atom in self.mol.GetAtoms():
            self.feat_mat[ind, :] = props.get(atom.GetAtomicNum())
            ind = ind + 1
        # self.to(device)

        # except Exception as e:
        #     print(f"Error processing SMILES: {smiles}")
        #     print(traceback.format_exc())  # 예외 정보를 출력
        #     # return None, None

        edges = adj_mat_to_edges(self.adj_mat)
        self.add_nodes(self.adj_mat.shape[0])
        if edges:
            src, dst = tuple(zip(*edges))
            self.add_edges(src, dst)

        self.ndata['feat'] = torch.tensor(self.feat_mat, dtype=torch.float32)#.to(device)
        
        node_id = 0
        for atom in self.mol.GetAtoms():
            self.atomic_nodes.append(atom.GetSymbol())
            self.neighbors[node_id] = atoms_to_symbols(atom.GetNeighbors())
            node_id += 1


    @classmethod
    def from_smiles(cls, smiles: str):
        g = cls(smiles)
        gg = g.to(device)
        return g.mol, g