import sys
from dataclasses import dataclass
from typing import Callable, Optional

from utils import mol_collate

import utils.mol_conv as mc

@dataclass(frozen=True)
class DatasetSpec:
    reader: Callable
    dataset_path: str
    num_desc: int
    collate_name: str
    # collate_module: str
    default_batch_size: None
    ckpt_path: Optional[str] = None

DATASET_REGISTRY = {
    "freesolv": DatasetSpec(
        reader = mc.read_dataset_freesolv,
        dataset_path='./datasets/freesolv',
        num_desc = 50, # 50
        collate_name = "descriptor_selection_freesolv",
        # collate_module = "mol_collate_freesolv",
        default_batch_size = None,
        ckpt_path='./checkpoints/freesolv/model_freesolv_300.pt'
    ),
    "esol": DatasetSpec(
        reader = mc.read_dataset_esol,
        dataset_path='./datasets/esol',
        num_desc = 63, # 63
        collate_name = "descriptor_selection_esol",
        # collate_module = "mol_collate_esol",
        default_batch_size = None,
        ckpt_path='./checkpoints/esol/model_esol_300.pt'
    ),
    "lipo": DatasetSpec(
        reader = mc.read_dataset_lipo,
        dataset_path='./datasets/lipo',
        num_desc = 25, # 25
        collate_name = "descriptor_selection_lipo",
        # collate_module = "mol_collate_esol",
        default_batch_size = 128,
        ckpt_path='./checkpoints/esol/model_esol_300.pt'
    ),
    "vp": DatasetSpec(
        reader = mc.read_dataset_vp,
        dataset_path='./datasets/vp',
        num_desc = 23,
        collate_name = "descriptor_selection_vp",
        # collate_module = "mol_collate_vp",
        default_batch_size = 128,
        ckpt_path='./checkpoints/vp/model_vp_300.pt'
    ),
    "solubility": DatasetSpec(
        reader = mc.read_dataset_solubility,
        dataset_path='./datasets/solubility',
        num_desc = 30,
        collate_name = "descriptor_selection_solubility",
        # collate_module = "mol_collate_solubility",
        default_batch_size = 256,
        ckpt_path='./checkpoints/solubility/model_solubility_300.pt'
    ),
    "freesolv_full": DatasetSpec(
        reader = mc.read_dataset_full,
        dataset_path='./datasets/freesolv',
        num_desc = 196,
        collate_name = "full_descriptors",
        default_batch_size = None,
        ckpt_path='./checkpoints/solubility/model_solubility_300.pt' # 수정필요
    ),
    "esol_full": DatasetSpec(
        reader = mc.read_dataset_full,
        dataset_path='./datasets/esol',
        num_desc = 196,
        collate_name = "full_descriptors",
        default_batch_size = None,
        ckpt_path='./checkpoints/solubility/model_solubility_300.pt' # 수정필요
    ),
    "vp_full": DatasetSpec(
        reader = mc.read_dataset_full,
        dataset_path='./datasets/vp',
        num_desc = 196,
        collate_name = "full_descriptors",
        default_batch_size = 128,
        ckpt_path='./checkpoints/solubility/model_solubility_300.pt' # 수정필요
    ),
    "solubility_full": DatasetSpec(
        reader = mc.read_dataset_full,
        dataset_path='./datasets/solubility',
        num_desc = 196,
        collate_name = "full_descriptors",
        default_batch_size = 256,
        ckpt_path='./checkpoints/solubility/model_solubility_300.pt' # 수정필요
    ),
    
}

def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    return DATASET_REGISTRY[name]

def build_collate_fn(dataset_name: str):
    spec = get_dataset_spec(dataset_name)
    collate_fn = getattr(mol_collate, spec.collate_name)

    return collate_fn

# def build_collate_fn(dataset_name: str, model: str):

#     spec = get_dataset_spec(dataset_name)
#     if model in ['GCN']:
#         collate_fn = getattr(mol_collate_newds, spec.collate_name)
#     elif model in ['gat']:
#         collate_fn = mol_collate.graph_only        

#     return collate_fn

def bs(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])