
import sys
from functools import partial
from dataclasses import dataclass
from typing import Callable, Optional

from utils.feat_map import build_feat_map
from utils import mol_collate
import utils.mol_conv_LapPE as mc

@dataclass(frozen=True)
class DatasetSpec:
    reader: Callable                        # ex) mc_new.read_dataset_freesolv
    num_desc_2d: int                        # 2D descriptor dim
    collate_name: str                       # ex) "collate_fusion_freesolv"
    collate_module: str                     # ex) "mol_collate_freesolv"
    default_batch_size: None

DATASET_REGISTRY = {
    "freesolv": DatasetSpec(
        reader = mc.read_dataset_freesolv,
        num_desc_2d = 50,
        collate_name = "collate_fusion_freesolv",
        collate_module = "mol_collate_freesolv",
        default_batch_size = None,
    ),
    "esol": DatasetSpec(
        reader = mc.read_dataset_esol,
        num_desc_2d = 63,
        collate_name = "collate_fusion_esol",
        collate_module = "mol_collate_esol",
        default_batch_size = None,
    ),
    "lipo": DatasetSpec(
        reader = mc.read_dataset_lipo,
        num_desc_2d = 25,
        collate_name = "collate_fusion_lipo",
        collate_module = "mol_collate_lipo",
        default_batch_size = 128,
    ),
    "scgas": DatasetSpec(
        reader = mc.read_dataset_scgas,
        num_desc_2d = 23,
        collate_name = "collate_fusion_scgas",
        collate_module = "mol_collate_scgas",
        default_batch_size = 128,
    ),
    "solubility": DatasetSpec(
        reader = mc.read_dataset_solubility,
        num_desc_2d = 30,
        collate_name = "collate_fusion_solubility",
        collate_module = "mol_collate_solubility",
        default_batch_size = 256,
    ),
}

def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    return DATASET_REGISTRY[name]

def build_collate_fn(dataset_name: str):
    spec = get_dataset_spec(dataset_name)

    feat3d_map, feat3d_cols = build_feat_map(dataset_name)
    num_desc_3d = len(feat3d_cols)

    collate_func = getattr(mol_collate, spec.collate_name)
    collate_fn = partial(collate_func, feat3d_map=feat3d_map, feat3d_dim=num_desc_3d)

    return collate_fn, num_desc_3d

def bs(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])