import numpy as np
import traceback
import dgl
import torch
from rdkit import Chem

from utils.utils import adj_mat_to_edges, atoms_to_symbols
from utils.mol_props import props

from scipy import sparse as sp

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

def laplacian_positional_encoding(adj_mat, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    n = adj_mat.shape[0]
    if n == 0:
        return torch.zeros((0, pos_enc_dim), dtype=torch.float32)

    A = sp.csr_matrix(adj_mat.astype(float))

    deg = np.asarray(A.sum(axis=1)).squeeze()
    deg = np.clip(deg, 1.0, None)
    D_inv_sqrt = sp.diags(deg ** -0.5, dtype=float)
    L = sp.eye(n, dtype=float) - (D_inv_sqrt @ A @ D_inv_sqrt)

    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVec = np.real(EigVec[:, idx])

    k = min(pos_enc_dim, max(n - 1, 0))
    pe = np.zeros((n, pos_enc_dim), dtype=np.float32)
    if k > 0:
        pe[:, :k] = EigVec[:, 1:1 + k]  # 첫번째(상수) 제외
    
    return torch.from_numpy(pe).float()

def construct_mol_graph(smiles, mol, adj_mat, feat_mat, pos_enc_dim=8):
    # molGraph = molDGLGraph(smiles, adj_mat, feat_mat, mol).to(device)

    edges = adj_mat_to_edges(adj_mat)
    src, dst = tuple(zip(*edges))
    molGraph = dgl.graph((src, dst), num_nodes=adj_mat.shape[0])

    # molGraph.add_nodes(adj_mat.shape[0])
    # molGraph.add_edges(src, dst)

    # Laplacian Positional Encoding
    lpe = laplacian_positional_encoding(adj_mat, pos_enc_dim)
    # molGraph.ndata['lap_pos_enc'] = molGraph.ndata['lap_pos_enc'].to(device)
    molGraph = molGraph.to(device)
    molGraph = dgl.add_self_loop(molGraph) # self loop
    molGraph.ndata['feat'] = torch.tensor(feat_mat, dtype=torch.float32, device=device)
    molGraph.ndata['lap_pos_enc'] = lpe.to(device)

    meta = {'smiles': smiles}

    return molGraph, meta

def smiles_to_mol_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        node_feat_mat = np.empty([mol.GetNumAtoms(), props.get(1).shape[0]])

        for i, atom in enumerate(mol.GetAtoms()):
            node_feat_mat[i, :] = props.get(atom.GetAtomicNum())

        g, meta = construct_mol_graph(smiles, mol, adj_mat, node_feat_mat)
        return mol, g, meta

    except Exception as e:
        print(f"Error processing SMILES: {smiles}")
        print(traceback.format_exc())

        return None, None, None
    
