import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold

# def get_scaffold_groups(smiles_list):
#     scaffold_dict = {}

#     for idx, s in enumerate(smiles_list):
#         scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s)

#         if scaffold not in scaffold_dict:
#             scaffold_dict[scaffold] = []

#         scaffold_dict[scaffold].append(idx)

#     print('scaffold_groups:', scaffold_dict)

#     return scaffold_dict


# def scaffold_info(smiles_list):
#     """
#     return
#         scaffold_ids : (N,)
#         num_scaffolds
#         scaffold_to_id
#     """
#     scaffold_dict = {}
#     scaffold_to_id = {}
#     scaffold_ids = []

#     for idx, s in enumerate(smiles_list):
#         scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s)

#         if scaffold not in scaffold_to_id:
#             scaffold_to_id[scaffold] = len(scaffold_to_id)

#         scaffold_ids.append(scaffold_to_id[scaffold])
#         if scaffold not in scaffold_dict:
#             scaffold_dict[scaffold] = []

#         scaffold_dict[scaffold].append(idx)

#     print('scaffold_groups:', scaffold_dict)

#     scaffold_ids = np.array(scaffold_ids)
#     num_scaffolds = len(scaffold_to_id)

#     return scaffold_dict, scaffold_ids, num_scaffolds, scaffold_to_id

def scaffold_info(smiles_list):
    """
    Returns
    -------
    scaffold_dict : dict
        scaffold string -> list of molecule indices
    scaffold_ids : np.ndarray of shape (N,)
        scaffold class id for each molecule
    num_scaffolds : int
        number of unique scaffolds
    scaffold_to_id : dict
        scaffold string -> integer id
    """
    scaffold_dict = {}
    scaffold_to_id = {}
    scaffold_ids = []

    for idx, s in enumerate(smiles_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s)

        if scaffold not in scaffold_to_id:
            scaffold_to_id[scaffold] = len(scaffold_to_id)
            scaffold_dict[scaffold] = []

        scaffold_ids.append(scaffold_to_id[scaffold])
        scaffold_dict[scaffold].append(idx)

    scaffold_ids = np.array(scaffold_ids, dtype=np.int64)
    num_scaffolds = len(scaffold_to_id)
    print('scaffold_groups:', scaffold_dict)

    return scaffold_dict, scaffold_ids, num_scaffolds, scaffold_to_id


def scaffold_split(smiles_list, train_size=0.8):
    scaffold_dict, _, _, _ = scaffold_info(smiles_list)

    groups = list(scaffold_dict.values())
    groups = sorted(groups, key=len, reverse=True)

    n_total = len(smiles_list)
    n_train_target = int(train_size * n_total)
    
    train_idx, test_idx = [], []

    for g in groups:
        if len(train_idx)+len(g) <= n_train_target:
            train_idx.extend(g)
        else:
            test_idx.extend(g)

    return np.array(train_idx), np.array(test_idx)

from rdkit.Chem.Scaffolds import MurckoScaffold




from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# def get_scaffold_id_map(smiles_list: list[str]):
#     """
#     SMILES 리스트로부터 Murcko scaffold 기반 정수 ID 매핑 생성.

#     Returns:
#         scaffold_ids  : list[int]  각 분자의 scaffold 정수 ID (-1: 파싱 실패)
#         n_scaffolds   : int        고유 scaffold 수
#     """
#     scaffold_to_id: dict[str, int] = {}
#     scaffold_ids:   list[int]      = []

#     for smi in smiles_list:
#         mol = Chem.MolFromSmiles(smi)
#         if mol is None:
#             scaffold_ids.append(-1)
#             continue
#         scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(
#             mol=mol, includeChirality=False
#         )
#         if scaffold_smi not in scaffold_to_id:
#             scaffold_to_id[scaffold_smi] = len(scaffold_to_id)
#         scaffold_ids.append(scaffold_to_id[scaffold_smi])

#     return scaffold_ids, len(scaffold_to_id)
