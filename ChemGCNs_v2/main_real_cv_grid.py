import random
import pandas as pd
from functools import partial
from itertools import product

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

import utils.mol_conv as mc
import utils.mol_conv_new as mc_new
from utils import cv_gridsearch
from utils import mol_collate, mol_collate_new
from utils.mol_props import dim_atomic_feat
from utils.feat_map import build_feat_map

from configs.config import SET_SEED, BACKBONE, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K, SEED

def main():
    SET_SEED()
    global BATCH_SIZE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if DATASET_NAME == 'freesolv':
        print('DATASET_NAME: ', DATASET_NAME)
        from utils.ablation import mol_collate_freesolv as mcol
        # dataset = mc.read_dataset_freesolv(DATASET_PATH + '.csv')
        dataset_new = mc_new.read_dataset_freesolv(DATASET_PATH + '.csv')
        num_descriptors_2d = 50

        # 임시 (scgas용 3d descriptor)
        feat3d_map, feat3d_cols = build_feat_map(DATASET_NAME)
        num_descriptors_3d = len(feat3d_cols)
        collate_fn = partial(mol_collate_new.collate_fusion_freesolv, feat3d_map=feat3d_map, feat3d_dim=num_descriptors_3d)

    elif DATASET_NAME == 'esol':
        print('DATASET_NAME: ', DATASET_NAME)
        from utils.ablation import mol_collate_esol as mcol
        # dataset = mc.read_dataset_esol(DATASET_PATH + '.csv')
        dataset_new = mc_new.read_dataset_esol(DATASET_PATH + '.csv')
        num_descriptors_2d = 63
        # descriptors = mol_collate.descriptor_selection_esol

        # 임시 (scgas용 3d descriptor)
        feat3d_map, feat3d_cols = build_feat_map(DATASET_NAME)
        num_descriptors_3d = len(feat3d_cols)
        collate_fn = partial(mol_collate_new.collate_fusion_esol, feat3d_map=feat3d_map, feat3d_dim=num_descriptors_3d)

    elif DATASET_NAME == 'lipo':
        print('DATASET_NAME: ', DATASET_NAME)
        from utils.ablation import mol_collate_lipo as mcol
        dataset = mc.read_dataset_lipo(DATASET_PATH + '.csv')
        num_descriptors_2d = 25
        descriptors = mol_collate.descriptor_selection_lipo

    elif DATASET_NAME == 'scgas':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 128
        # from utils.ablation import mol_collate_scgas as mcol
        # dataset = mc.read_dataset_scgas(DATASET_PATH + '.csv')
        dataset_new = mc_new.read_dataset_scgas(DATASET_PATH + '.csv')
        num_descriptors_2d = 23
        # descriptors = mol_collate.descriptor_selection_scgas

        feat3d_map, feat3d_cols = build_feat_map(DATASET_NAME)
        num_descriptors_3d = len(feat3d_cols)
        collate_fn = partial(mol_collate_new.collate_fusion_scgas, feat3d_map=feat3d_map, feat3d_dim=num_descriptors_3d)

    elif DATASET_NAME == 'solubility':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 256
        from utils.ablation import mol_collate_solubility as mcol
        # dataset = mc.read_dataset_solubility(DATASET_PATH + '.csv')
        dataset_new = mc_new.read_dataset_solubility(DATASET_PATH + '.csv')
        num_descriptors_2d = 30
        # descriptors = mol_collate.descriptor_selection_solubility

        # 임시 (scgas용 3d descriptor)
        feat3d_map, feat3d_cols = build_feat_map(DATASET_NAME)
        num_descriptors_3d = len(feat3d_cols)
        collate_fn = partial(mol_collate_new.collate_fusion_solubility, feat3d_map=feat3d_map, feat3d_dim=num_descriptors_3d)

    # random.shuffle(dataset)
    # random.shuffle(dataset_new)
    train_dataset, test_dataset = train_test_split(dataset_new, test_size = 0.2, random_state = SEED)

    if BACKBONE == 'GCN':
        from model import KROVEX_baseline, KROVEX_GCNs
        KROVEX = KROVEX_baseline.Net(dim_atomic_feat, num_descriptors_2d).to(device)
        KROVEX_GCNs = KROVEX_GCNs.Net(dim_atomic_feat, num_descriptors_2d).to(device)

        from model import CrossAttn_TFN
        md_CrossAttn_TFN = CrossAttn_TFN.Net_2d(dim_atomic_feat, num_descriptors_2d, num_descriptors_3d).to(device)


    # loss function
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    val_losses = dict()

    # -------------------------- TEST GRID SEARCH ------------------------------ #
    grid = {
        "d_t": [32, 64],
        "d_k": [16, 32, 64],
        "fc1": [128, 256],
        "fc2": [32, 64],
        "dropout": [0.1, 0.3],
    }

    param_list = [
        dict(zip(grid.keys(), vals))
        for vals in product(*grid.values())
    ]
    print("num configs:", len(param_list))
    # --------------------------  ------------------------------ #


    print(f'{BACKBONE}, {DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')

    # -------------------------- Baseline ------------------------------ #
    # # KROVEX
    # test_losses['KROVEX'], test_losses['KROVEX_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, KROVEX, criterion, BATCH_SIZE, MAX_EPOCHS, collate_fn, model_name='KROVEX')
    # print(f'Final test | loss: ' + str(test_losses['KROVEX']) + '| R2: ' + str(test_losses['KROVEX_R2']))
    
    # # KROVEX GCN 직접 구현
    # test_losses['KROVEX_GCNs'], test_losses['KROVEX_GCNs_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, KROVEX_GCNs, criterion, BATCH_SIZE, MAX_EPOCHS, collate_fn, model_name='KROVEX_GCNs')
    # print(f'Final test | loss: ' + str(test_losses['KROVEX_GCNs']) + '| R2: ' + str(test_losses['KROVEX_GCNs_R2']))
    # ------------------------------------------------------------------ #

    # -------------------------- Cross Attention + TFN ------------------------------ #
    val_losses['md_CrossAttn_TFN'] = cv_gridsearch.grid_search_kfold(train_dataset, dim_atomic_feat, num_descriptors_2d, num_descriptors_3d, param_list, criterion, K, BATCH_SIZE, MAX_EPOCHS, collate_fn, model_name = 'md_CrossAttn_TFN')

    # ------------------------------------------------------------------ #

    print(f'{BACKBONE}, {DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')


if __name__ == '__main__':
    main()