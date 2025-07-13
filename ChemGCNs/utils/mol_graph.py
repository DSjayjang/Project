import numpy as np
import traceback
import dgl
import torch
from rdkit import Chem

from utils.utils import adj_mat_to_edges, atoms_to_symbols
from utils.mol_props import props

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def construct_mol_graph(smiles, mol, adj_mat, feat_mat):
    molGraph = molDGLGraph(smiles, adj_mat, feat_mat, mol).to(device)
    # molGraph = MolGraph(smiles).to(device)
    edges = adj_mat_to_edges(adj_mat)
    src, dst = tuple(zip(*edges))

    molGraph.add_nodes(adj_mat.shape[0])
    molGraph.add_edges(src, dst)
    molGraph.ndata['feat'] = torch.tensor(feat_mat, dtype=torch.float32).to(device)

    return molGraph

def smiles_to_mol_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        node_feat_mat = np.empty([mol.GetNumAtoms(), props.get(1).shape[0]])

        ind = 0
        for atom in mol.GetAtoms():
            node_feat_mat[ind, :] = props.get(atom.GetAtomicNum())
            ind = ind + 1

        return mol, construct_mol_graph(smiles, mol, adj_mat, node_feat_mat)
    except Exception as e:
        print(f"Error processing SMILES: {smiles}")
        print(traceback.format_exc())  # 예외 정보를 출력
        return None, None

class MolGraph(dgl.DGLGraph):
    """
    SMILES -> RDKit Mol -> DGLGraph 빌드 클래스
    노드 속성으로는 atomic_props 에서 가져온 z-score 표준화된 벡터를 사용.
    """
    def __init__(self, smiles):
        super(MolGraph, self).__init__()
        self.smiles = smiles
        self.mol = None
        self.adj_mat = None
        self.feat_mat = None
        self._build_graph()

    def _build_graph(self):
        try:
            self.mol = Chem.MolFromSmiles(self.smiles)
            self.adj_mat = Chem.GetAdjacencyMatrix(self.mol)
            
            n_nodes = self.adj_mat.shape[0]
            feat_dim = len(next(iter(props.values())))
            self.feat_mat = np.zeros([n_nodes, feat_dim], dtype = float)
        
            # Node features
            for idx, atom in enumerate(self.mol.GetAtoms()):
                self.feat_mat[idx, :] = props.get(atom.GetAtomicNum())

            # Build DGL graph
            self.add_nodes(n_nodes)
            edges = adj_mat_to_edges(self.adj_mat)
            if edges:
                src, dst = zip(*edges)
                self.add_edges(src, dst)

            # Assign node features
            self.ndata['feat'] = torch.tensor(self.feat_mat, dtype = torch.float32)
            self.to(device)

        except Exception as e:
            print(f'Error processing SMILES: {self.smiles}')
            print(traceback.format_exc())
            self.mol = None
            self.adj_mat = None
            self.feat_mat = None

    def samples(self):
        return self.samples