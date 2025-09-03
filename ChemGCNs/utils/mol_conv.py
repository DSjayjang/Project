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