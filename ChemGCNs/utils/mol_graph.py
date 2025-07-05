import numpy as np
import traceback

import dgl
import torch
from rdkit import Chem

from utils.mol_props import props
from utils.utils import adj_mat_to_edges

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MolGraph(dgl.DGLGraph):
    """
    SMILES -> RDKit Mol -> DGLGraph 빌드 클래스
    노드 속성으로는 atomic_props 에서 가져온 z-score 표준화된 벡터를 사용.
    """
    # def __init__(self, smiles, graph_device = None):
    def __init__(self, smiles):
        super(MolGraph, self).__init__()
        self.smiles = smiles
        self.mol = None
        self.adj_mat = None
        self.feat_mat = None
        # self.graph_device = graph_device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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