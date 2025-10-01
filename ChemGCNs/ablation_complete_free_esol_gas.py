import random

import torch
import torch.nn as nn

from utils.mol_props import dim_atomic_feat

import utils.mol_conv as mc
from utils import trainer
from utils import mol_collate

from model import CONCAT_DS, KROVEX
from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K

def main():
    SET_SEED()

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

    elif DATASET_NAME == 'scgas':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 128
        from utils.ablation import mol_collate_scgas as mcol
        dataset = mc.read_dataset_esol(DATASET_PATH + '.csv')
        num_descriptors = 23
        descriptors = mol_collate.descriptor_selection_scgas

    random.shuffle(dataset)

    # concatenation + descriptor selection
    model_concat_3 = CONCAT_DS.concat_Net_3(dim_atomic_feat, 1, 3).to(device)
    model_concat_5 = CONCAT_DS.concat_Net_5(dim_atomic_feat, 1, 5).to(device)
    model_concat_7 = CONCAT_DS.concat_Net_7(dim_atomic_feat, 1, 7).to(device)
    model_concat_10 = CONCAT_DS.concat_Net_10(dim_atomic_feat, 1, 10).to(device)
    model_concat_20 = CONCAT_DS.concat_Net_20(dim_atomic_feat, 1, 20).to(device)
    model_concat_ds = CONCAT_DS.concat_Net(dim_atomic_feat, 1, num_descriptors).to(device)

    # kronecker-product + descriptor selection
    model_kronecker_3 = KROVEX.kronecker_Net_3(dim_atomic_feat, 1, 3).to(device)
    model_kronecker_5 = KROVEX.kronecker_Net_5(dim_atomic_feat, 1, 5).to(device)
    model_kronecker_7 = KROVEX.kronecker_Net_7(dim_atomic_feat, 1, 7).to(device)
    model_kronecker_10 = KROVEX.kronecker_Net_10(dim_atomic_feat, 1, 10).to(device)
    model_kronecker_20 = KROVEX.kronecker_Net_20(dim_atomic_feat, 1, 20).to(device)
    model_KROVEX = KROVEX.Net(dim_atomic_feat, 1, num_descriptors).to(device)

    # loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()
    
    # ------------------------ concatenation + descriptor selection ------------------------#
    print('--------- concatenation with 3 descriptors ---------')
    test_losses['concat_3'] = trainer.cross_validation(dataset, model_concat_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_3)
    print('test loss (concat_3): ' + str(test_losses['concat_3']))

    print('--------- concatenation with 5 descriptors ---------')
    test_losses['concat_5'] = trainer.cross_validation(dataset, model_concat_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_5)
    print('test loss (concat_5): ' + str(test_losses['concat_5']))

    print('--------- concatenation with 7 descriptors ---------')
    test_losses['concat_7'] = trainer.cross_validation(dataset, model_concat_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_7)
    print('test loss (concat_7): ' + str(test_losses['concat_7']))

    print('--------- concatenation with 10 descriptors ---------')
    test_losses['concat_10'] = trainer.cross_validation(dataset, model_concat_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_10)
    print('test loss (concat_10): ' + str(test_losses['concat_10']))

    print('--------- concatenation with 20 descriptors ---------')
    test_losses['concat_20'] = trainer.cross_validation(dataset, model_concat_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_20)
    print('test loss (concat_20): ' + str(test_losses['concat_20']))

    print('--------- concatenation with descriptor selection ---------')
    test_losses['concat_ds'] = trainer.cross_validation(dataset, model_concat_ds, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, descriptors)
    print('test loss (concat_ds): ' + str(test_losses['concat_ds']))

    #------------------------ kronecker-product + descriptor selection ------------------------#
    print('--------- kronecker-product with 3 descriptors ---------')
    test_losses['kronecker_3'] = trainer.cross_validation(dataset, model_kronecker_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_3)
    print('test loss (kronecker_3): ' + str(test_losses['kronecker_3']))

    print('--------- kronecker-product with 5 descriptors ---------')
    test_losses['kronecker_5'] = trainer.cross_validation(dataset, model_kronecker_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_5)
    print('test loss (kronecker_5): ' + str(test_losses['kronecker_5']))

    print('--------- kronecker-product with 7 descriptors ---------')
    test_losses['kronecker_7'] = trainer.cross_validation(dataset, model_kronecker_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_7)
    print('test loss (kronecker_7): ' + str(test_losses['kronecker_7']))

    print('--------- kronecker-product with 10 descriptors ---------')
    test_losses['kronecker_10'] = trainer.cross_validation(dataset, model_kronecker_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_10)
    print('test loss (kronecker_10): ' + str(test_losses['kronecker_10']))

    print('--------- kronecker-product with 20 descriptors ---------')
    test_losses['kronecker_20'] = trainer.cross_validation(dataset, model_kronecker_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_20)
    print('test loss (kronecker_20): ' + str(test_losses['kronecker_20']))

    print('--------- kronecker-product with descriptor selection ---------')
    test_losses['KROVEX'] = trainer.cross_validation(dataset, model_KROVEX, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, descriptors)
    print('test loss (KROVEX): ' + str(test_losses['KROVEX']))

    print('test_losse:', test_losses)
    print(DATASET_NAME)

if __name__ == '__main__':
    main()