import os
import pandas as pd
import numpy as np
import rdkit.Chem.Descriptors as dsc

from utils.utils import FeatureNormalization
from utils.mol_graph import MolGraph

class MoleculeDataset:
    """
    내용 확인
    Reads a CSV file of SMILES and targets, constructs DGL graphs with atomic and molecular features,
    and applies z-score normalization to self-features.
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.samples = []
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

            graph_obj.num_atoms = graph_obj.mol.GetNumAtoms()
            graph_obj.weight = dsc.ExactMolWt(graph_obj.mol)
            graph_obj.num_rings = graph_obj.mol.GetRingInfo().NumRings()
            graph_obj.max_abs_charge = dsc.MaxAbsPartialCharge(graph_obj.mol)
            graph_obj.min_abs_charge = dsc.MinAbsPartialCharge(graph_obj.mol)
            graph_obj.num_rad_elc = dsc.NumValenceElectrons(graph_obj.mol)
            graph_obj.num_val_elc = dsc.NumValenceElectrons(graph_obj.mol)
            
            self.samples.append((graph_obj, tgt))
            self.graphs.append(graph_obj)

        for feat in ['num_atoms', 'weight', 'num_rings', 'max_abs_charge', 'min_abs_charge', 'num_rad_elc', 'num_val_elc']:
            FeatureNormalization(self.graphs, feat)

        # for idx, smile in enumerate(smiles):
        #     mol, mol_graph = smiles_to_mol_graph(smile)

        #     # Add molecular descriptors as features
        #     if mol is not None and mol_graph is not None:
        #         mol_graph.num_atoms = mol.GetNumAtoms()
        #         mol_graph.num_atoms = mol.GetNumAtoms()
        #         mol_graph.weight = dsc.ExactMolWt(mol)
        #         mol_graph.num_rings = mol.GetRingInfo().NumRings()
        #         mol_graph.max_abs_charge = dsc.MaxAbsPartialCharge(mol)
        #         mol_graph.min_abs_charge = dsc.MinAbsPartialCharge(mol)
        #         mol_graph.num_rad_elc = dsc.NumValenceElectrons(mol) # (원본) 잘못된 것
        #         mol_graph.num_val_elc = dsc.NumValenceElectrons(mol)

        #         # Collect samples and graph list
        #         self.samples.append((mol_graph, targets[idx]))
        #         self.mol_graphs.append(mol_graph)
            
            # Normalize each features across all graphs
            # for feat_name in ['num_atoms', 'weight', 'num_rings', 'max_abs_charge', 'min_abs_charge', 'num_rad_elc', 'num_val_elc']:
            #     utils.FeatureNormalization(self.mol_graphs, feat_name)
    
    def __len__(self):
        return (len(self.samples))