import random
import torch
import torch.nn as nn

import utils.mol_conv as mc
from utils import trainer, trainer_bilinear
from utils import mol_collate_bilinear
from utils.mol_props import dim_atomic_feat

from model import Bilinear_Attn, KROVEX, CONCAT_DS
from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K, SEED

def main():
    SET_SEED()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if DATASET_NAME == 'freesolv':
        print('DATASET_NAME: ', DATASET_NAME)
        from utils.ablation import mol_collate_freesolv as mcol
        dataset = mol_collate_bilinear.read_dataset_freesolv_bilinear(DATASET_PATH + '.csv')
        num_descriptors = 208
        descriptors = mol_collate_bilinear.descriptor_selection_freesolv_bilinear

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
        global BATCH_SIZE
        BATCH_SIZE = 128
        from utils.ablation import mol_collate_scgas as mcol
        dataset = mc.read_dataset_scgas(DATASET_PATH + '.csv')
        num_descriptors = 23
        descriptors = mol_collate.descriptor_selection_scgas

    elif DATASET_NAME == 'solubility':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 256
        from utils.ablation import mol_collate_solubility as mcol
        dataset = mc.read_dataset_solubility(DATASET_PATH + '.csv')
        num_descriptors = 16
        descriptors = mol_collate.descriptor_selection_solubility

    random.shuffle(dataset)

    # bilinear attention + descriptor selection
    # model_bilinear_attn_3 = Bilinear_Attn.Net(dim_atomic_feat, 1, 3).to(device)
    # model_bilinear_attn_5 = Bilinear_Attn.Net(dim_atomic_feat, 1, 5).to(device)
    # model_bilinear_attn_7 = Bilinear_Attn.Net(dim_atomic_feat, 1, 7).to(device)
    # model_bilinear_attn_10 = Bilinear_Attn.Net(dim_atomic_feat, 1, 10).to(device)
    # model_bilinear_attn_20 = Bilinear_Attn.Net(dim_atomic_feat, 1, 20).to(device)
    model_bilinear_attn_ds = Bilinear_Attn.Net(dim_atomic_feat, 1, num_descriptors).to(device)

    # loss function
    # criterion = nn.L1Loss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()

    print(f'{DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')

    # ------------------------ bilinear attention + descriptor selection ------------------------#
    # print('--------- bilinear attention with 3 descriptors ---------')
    # test_losses['bilinear_attn_3'] = trainer_bilinear.cross_validation(dataset, model_bilinear_attn_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_bilinear.train_model, trainer_bilinear.test_model, mcol.descriptor_selection_3)
    # print('test loss (bilinear_attn_3): ' + str(test_losses['bilinear_attn_3']))

    # print('--------- bilinear attention with 5 descriptors ---------')
    # test_losses['bilinear_attn_5'] = trainer_bilinear.cross_validation(dataset, model_bilinear_attn_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_bilinear.train_model, trainer_bilinear.test_model, mcol.descriptor_selection_5)
    # print('test loss (bilinear_attn_5): ' + str(test_losses['bilinear_attn_5']))

    # print('--------- bilinear attention with 7 descriptors ---------')
    # test_losses['bilinear_attn_7'] = trainer_bilinear.cross_validation(dataset, model_bilinear_attn_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_bilinear.train_model, trainer_bilinear.test_model, mcol.descriptor_selection_7)
    # print('test loss (bilinear_attn_7): ' + str(test_losses['bilinear_attn_7']))

    # print('--------- bilinear attention with 10 descriptors ---------')
    # test_losses['bilinear_attn_10'] = trainer_bilinear.cross_validation(dataset, model_bilinear_attn_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_bilinear.train_model, trainer_bilinear.test_model, mcol.descriptor_selection_10)
    # print('test loss (bilinear_attn_10): ' + str(test_losses['bilinear_attn_10']))

    # print('--------- bilinear attention with 20 descriptors ---------')
    # test_losses['bilinear_attn_20'] = trainer_bilinear.cross_validation(dataset, model_bilinear_attn_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_bilinear.train_model, trainer_bilinear.test_model, mcol.descriptor_selection_20)
    # print('test loss (bilinear_attn_20): ' + str(test_losses['bilinear_attn_20']))

    print('--------- bilinear attention with descriptor selection ---------')
    test_losses['bilinear_attn_ds'] = trainer_bilinear.cross_validation(dataset, model_bilinear_attn_ds, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer_bilinear.train_model, trainer_bilinear.test_model, descriptors)
    print('test loss (bilinear_attn_ds): ' + str(test_losses['bilinear_attn_ds']))

    total_params = sum(p.numel() for p in model_bilinear_attn_ds.parameters() if p.requires_grad)
    print(f"bilinear attn 총 학습 가능한 파라미터 수: {total_params:,}")

    model_concat_ds = CONCAT_DS.concat_Net(dim_atomic_feat, 1, num_descriptors).to(device)
    model_KROVEX = KROVEX.Net(dim_atomic_feat, 1, num_descriptors).to(device)

    total_params = sum(p.numel() for p in model_concat_ds.parameters() if p.requires_grad)
    print(f"concat ds 총 학습 가능한 파라미터 수: {total_params:,}")
    total_params = sum(p.numel() for p in model_KROVEX.parameters() if p.requires_grad)
    print(f"krovex 총 학습 가능한 파라미터 수: {total_params:,}")

    print('--------- bilinear attention with descriptor selection ---------')
    test_losses['model_KROVEX'] = trainer.cross_validation(dataset, model_KROVEX, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, descriptors)
    print('test loss (model_KROVEX): ' + str(test_losses['model_KROVEX']))

    print('test_losse:', test_losses)
    print(f'{DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')

if __name__ == '__main__':
    main()