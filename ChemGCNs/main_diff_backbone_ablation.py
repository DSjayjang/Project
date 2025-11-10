import random
import torch
import torch.nn as nn

import utils.mol_conv as mc
from utils import trainer
from utils import mol_collate
from utils.mol_props import dim_atomic_feat


from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K

backbone = 'GAT' # ['GAT', ]

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

    random.shuffle(dataset)

    if backbone == 'GAT':
        # GAT + kronecker-product + descriptor selection
        from model import GAT_Fusion
        model_kronecker_3 = GAT_Fusion.kronecker_3(dim_atomic_feat, 1, 4, 3).to(device)
        model_kronecker_5 = GAT_Fusion.kronecker_5(dim_atomic_feat, 1, 4, 5).to(device)
        model_kronecker_7 = GAT_Fusion.kronecker_7(dim_atomic_feat, 1, 4, 7).to(device)
        model_kronecker_10 = GAT_Fusion.kronecker_10(dim_atomic_feat, 1, 4, 10).to(device)
        model_kronecker_20 = GAT_Fusion.kronecker_20(dim_atomic_feat, 1, 4, 20).to(device)
        model_Fusion = GAT_Fusion.Net(dim_atomic_feat, 1, 4, num_descriptors).to(device)
        
    else:
        print('다시')


    # loss function
    # criterion = nn.L1Loss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()
    
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
    test_losses['Backbone_Fusion'] = trainer.cross_validation(dataset, model_Fusion, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, descriptors)
    print('test loss (Backbone_Fusion): ' + str(test_losses['Backbone_Fusion']))

    print('test_losse:', test_losses)
    print(backbone, DATASET_NAME)

if __name__ == '__main__':
    main()