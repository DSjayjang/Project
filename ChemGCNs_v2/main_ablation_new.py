import random
import pandas as pd
from functools import partial
import torch
import torch.nn as nn

import utils.mol_conv as mc
import utils.mol_conv_new as mc_new
from utils import trainer, trainer_new
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
        dataset = mc.read_dataset_freesolv(DATASET_PATH + '.csv')
        num_descriptors = 50
        descriptors = mol_collate.descriptor_selection_freesolv

    elif DATASET_NAME == 'esol':
        print('DATASET_NAME: ', DATASET_NAME)
        from utils.ablation import mol_collate_esol as mcol
        dataset = mc.read_dataset_esol(DATASET_PATH + '.csv')
        num_descriptors = 63
        descriptors = mol_collate.descriptor_selection_esol

    elif DATASET_NAME == 'lipo':
        print('DATASET_NAME: ', DATASET_NAME)
        from utils.ablation import mol_collate_lipo as mcol
        dataset = mc.read_dataset_lipo(DATASET_PATH + '.csv')
        num_descriptors = 25
        descriptors = mol_collate.descriptor_selection_lipo

    elif DATASET_NAME == 'scgas':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 128
        from utils.ablation import mol_collate_scgas as mcol
        dataset = mc.read_dataset_scgas(DATASET_PATH + '.csv')
        dataset_new = mc_new.read_dataset_scgas(DATASET_PATH + '.csv')
        num_descriptors = 23
        descriptors = mol_collate.descriptor_selection_scgas

        feat3d_map, feat3d_cols = build_feat_map(DATASET_NAME)
        num_descriptors_3d = len(feat3d_cols)
        collate_fn = partial(mol_collate_new.collate_fusion_scgas, feat3d_map=feat3d_map, feat3d_dim=num_descriptors_3d)

    elif DATASET_NAME == 'solubility':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 256
        from utils.ablation import mol_collate_solubility as mcol
        dataset = mc.read_dataset_solubility(DATASET_PATH + '.csv')
        num_descriptors = 30
        descriptors = mol_collate.descriptor_selection_solubility

    random.shuffle(dataset)
    random.shuffle(dataset_new)

    if BACKBONE == 'GCN':
        # GCN + kronecker-product + descriptor selection
        from model import KROVEX, KROVEX_new, TFN, Cross_Attn_TFN, Cross_Attn_TFN2, Cross_Attn_TFN3, Cross_Attn_TFN4
        
        model_Fusion = KROVEX.Net(dim_atomic_feat, 1, num_descriptors).to(device)
        model_Fusion_new = KROVEX_new.Net(dim_atomic_feat, 1, num_descriptors, num_descriptors_3d).to(device)
        model_TFN = TFN.Net(dim_atomic_feat, 1, num_descriptors, num_descriptors_3d).to(device)
        model_CATFN = Cross_Attn_TFN.Net(dim_atomic_feat, 1, num_descriptors, num_descriptors_3d).to(device)
        model_CATFN2 = Cross_Attn_TFN2.Net(dim_atomic_feat, 1, num_descriptors, num_descriptors_3d).to(device)
        model_CATFN3 = Cross_Attn_TFN3.Net(dim_atomic_feat, 1, num_descriptors, num_descriptors_3d).to(device)
        model_CATFN4 = Cross_Attn_TFN4.Net(dim_atomic_feat, 1, num_descriptors, num_descriptors_3d).to(device)
      
    elif BACKBONE == 'GAT':
        from model import GAT
        from utils import mol_collate_vanilla
        dataset_backbone = mc.read_dataset(DATASET_PATH + '.csv')
        random.shuffle(dataset_backbone)

        model_backbone = GAT.Net(dim_atomic_feat, 1, 4).to(device)
        
        # EGAT
        from model import EGAT
        model_backbone_R = EGAT.Net(dim_atomic_feat, 1, 4, 1).to(device)
        model_backbone_S = EGAT.Net(dim_atomic_feat, 1, 4, 2).to(device)
        model_backbone_E = EGAT.Net(dim_atomic_feat, 1, 4, 3).to(device)

        # GAT + concatenation + descriptor selection
        from model import GAT_CONCAT_DS
        model_concat_3 = GAT_CONCAT_DS.concat_3(dim_atomic_feat, 1, 4, 3).to(device)
        model_concat_5 = GAT_CONCAT_DS.concat_5(dim_atomic_feat, 1, 4, 5).to(device)
        model_concat_7 = GAT_CONCAT_DS.concat_7(dim_atomic_feat, 1, 4, 7).to(device)
        model_concat_10 = GAT_CONCAT_DS.concat_10(dim_atomic_feat, 1, 4, 10).to(device)
        model_concat_20 = GAT_CONCAT_DS.concat_20(dim_atomic_feat, 1, 4, 20).to(device)
        model_concat_ds = GAT_CONCAT_DS.concat_Net(dim_atomic_feat, 1, 4, num_descriptors).to(device)

        # GAT + kronecker-product + descriptor selection
        from model import GAT_Fusion
        model_kronecker_3 = GAT_Fusion.kronecker_3(dim_atomic_feat, 1, 4, 3).to(device)
        model_kronecker_5 = GAT_Fusion.kronecker_5(dim_atomic_feat, 1, 4, 5).to(device)
        model_kronecker_7 = GAT_Fusion.kronecker_7(dim_atomic_feat, 1, 4, 7).to(device)
        model_kronecker_10 = GAT_Fusion.kronecker_10(dim_atomic_feat, 1, 4, 10).to(device)
        model_kronecker_20 = GAT_Fusion.kronecker_20(dim_atomic_feat, 1, 4, 20).to(device)
        model_Fusion = GAT_Fusion.Net(dim_atomic_feat, 1, 4, num_descriptors).to(device)
    
    elif BACKBONE == 'GIN':
        from model import GIN
        from utils import mol_collate_vanilla
        dataset_backbone = mc.read_dataset(DATASET_PATH + '.csv')
        random.shuffle(dataset_backbone)

        model_backbone = GIN.Net(dim_atomic_feat, 1).to(device)

        # EGIN
        from model import EGIN
        model_backbone_R = EGIN.Net(dim_atomic_feat, 1, 1).to(device)
        model_backbone_S = EGIN.Net(dim_atomic_feat, 1, 2).to(device)
        model_backbone_E = EGIN.Net(dim_atomic_feat, 1, 3).to(device)

        # GIN + concatenation + descriptor selection
        from model import GIN_CONCAT_DS
        model_concat_3 = GIN_CONCAT_DS.concat_Net_3(dim_atomic_feat, 1, 3).to(device)
        model_concat_5 = GIN_CONCAT_DS.concat_Net_5(dim_atomic_feat, 1, 5).to(device)
        model_concat_7 = GIN_CONCAT_DS.concat_Net_7(dim_atomic_feat, 1, 7).to(device)
        model_concat_10 = GIN_CONCAT_DS.concat_Net_10(dim_atomic_feat, 1, 10).to(device)
        model_concat_20 = GIN_CONCAT_DS.concat_Net_20(dim_atomic_feat, 1, 20).to(device)
        model_concat_ds = GIN_CONCAT_DS.Net(dim_atomic_feat, 1, num_descriptors).to(device)

        # GIN + kronecker-product + descriptor selection
        from model import GIN_Fusion
        model_kronecker_3 = GIN_Fusion.kronecker_Net_3(dim_atomic_feat, 1, 3).to(device)
        model_kronecker_5 = GIN_Fusion.kronecker_Net_5(dim_atomic_feat, 1, 5).to(device)
        model_kronecker_7 = GIN_Fusion.kronecker_Net_7(dim_atomic_feat, 1, 7).to(device)
        model_kronecker_10 = GIN_Fusion.kronecker_Net_10(dim_atomic_feat, 1, 10).to(device)
        model_kronecker_20 = GIN_Fusion.kronecker_Net_20(dim_atomic_feat, 1, 20).to(device)
        model_Fusion = GIN_Fusion.Net(dim_atomic_feat, 1, num_descriptors).to(device)
    else:
        print('모델 정의 안됨')


    # loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()

    print(f'{BACKBONE}, {DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')

    # # krovex
    # test_losses['Backbone_Fusion'] = trainer.cross_validation(dataset, model_Fusion, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, descriptors)
    # print('test loss (Backbone_Fusion): ' + str(test_losses['Backbone_Fusion']))

    # # krovecx + 3d descriptor
    # test_losses['Backbone_Fusion_new'] = trainer_new.cross_validation(dataset_new, model_Fusion_new, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_new.train_model, trainer_new.test_model, collate_fn)
    # print('test loss (Backbone_Fusion_new): ' + str(test_losses['Backbone_Fusion_new']))

    # # tensor fusion
    # test_losses['Backbone_Tensor_Fusion'] = trainer_new.cross_validation(dataset_new, model_TFN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_new.train_model, trainer_new.test_model, collate_fn)
    # print('test loss (Backbone_Tensor_Fusion): ' + str(test_losses['Backbone_Tensor_Fusion']))

    # # cross attention + TFN
    # test_losses['Backbone_Cross_Attn_TFN'] = trainer_new.cross_validation(dataset_new, model_CATFN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_new.train_model, trainer_new.test_model, collate_fn)
    # print('test loss (Backbone_Cross_Attn_TFN): ' + str(test_losses['Backbone_Cross_Attn_TFN']))

    # # cross attention + TFN2
    # test_losses['Backbone_Cross_Attn_TFN2'] = trainer_new.cross_validation(dataset_new, model_CATFN2, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_new.train_model, trainer_new.test_model, collate_fn)
    # print('test loss (Backbone_Cross_Attn_TFN2): ' + str(test_losses['Backbone_Cross_Attn_TFN2']))

    # cross attention + TFN3
    test_losses['Backbone_Cross_Attn_TFN3'] = trainer_new.cross_validation(dataset_new, model_CATFN3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_new.train_model, trainer_new.test_model, collate_fn)
    print('test loss (Backbone_Cross_Attn_TFN3): ' + str(test_losses['Backbone_Cross_Attn_TFN3']))

    # cross attention + TFN4
    test_losses['Backbone_Cross_Attn_TFN4'] = trainer_new.cross_validation(dataset_new, model_CATFN4, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_new.train_model, trainer_new.test_model, collate_fn)
    print('test loss (Backbone_Cross_Attn_TFN4): ' + str(test_losses['Backbone_Cross_Attn_TFN4']))

    print('test_losse:', test_losses)
    print(f'{BACKBONE}, {DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')


if __name__ == '__main__':
    main()