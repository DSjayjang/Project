import random
import torch
import torch.nn as nn

import utils.mol_conv as mc
from utils import trainer
from utils import mol_collate
from utils.mol_props import dim_atomic_feat


from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K

backbone = 'GAT' # [GAT, SAGE, GIN]

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
        # EGAT
        from model import EGAT
        from utils import mol_collate_gcn
        dataset_backbone = mc.read_dataset(DATASET_PATH + '.csv')
        random.shuffle(dataset_backbone)
        model_backbone_R = EGAT.Net(dim_atomic_feat, 1, 4, 1).to(device)
        model_backbone_S = EGAT.Net(dim_atomic_feat, 1, 4, 2).to(device)
        model_backbone = EGAT.Net(dim_atomic_feat, 1, 4, 3).to(device)

        # GAT + concatenation + descriptor selection
        from model import GAT_CONCAT_DS
        model_concat_3 = GAT_CONCAT_DS.concat_3(dim_atomic_feat, 1, 4, 3).to(device)
        model_concat_5 = GAT_CONCAT_DS.concat_5(dim_atomic_feat, 1, 4, 5).to(device)
        model_concat_7 = GAT_CONCAT_DS.concat_7(dim_atomic_feat, 1, 4, 7).to(device)
        model_concat_10 = GAT_CONCAT_DS.concat_10(dim_atomic_feat, 1, 4, 10).to(device)
        model_concat_20 = GAT_CONCAT_DS.concat_20(dim_atomic_feat, 1, 4, 20).to(device)
        model_concat_ds = GAT_CONCAT_DS.concat_Net(dim_atomic_feat, 1, 4, num_descriptors).to(device)

        # # GAT + kronecker-product + descriptor selection
        # from model import GAT_Fusion
        # model_kronecker_3 = GAT_Fusion.kronecker_3(dim_atomic_feat, 1, 4, 3).to(device)
        # model_kronecker_5 = GAT_Fusion.kronecker_5(dim_atomic_feat, 1, 4, 5).to(device)
        # model_kronecker_7 = GAT_Fusion.kronecker_7(dim_atomic_feat, 1, 4, 7).to(device)
        # model_kronecker_10 = GAT_Fusion.kronecker_10(dim_atomic_feat, 1, 4, 10).to(device)
        # model_kronecker_20 = GAT_Fusion.kronecker_20(dim_atomic_feat, 1, 4, 20).to(device)
        # model_Fusion = GAT_Fusion.Net(dim_atomic_feat, 1, 4, num_descriptors).to(device)
        
    else:
        print('아직 모델 정의 안됨')


    # loss function
    # criterion = nn.L1Loss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()

    #------------------------ Backbone ------------------------#
    print('--------- Backbone with predefined descriptor Ring ---------')
    test_losses['Backbone_R'] = trainer.cross_validation(dataset_backbone, model_backbone_R, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mol_collate_gcn.collate_egcn_ring)
    print('test loss (Backbone_R): ' + str(test_losses['Backbone_R']))

    print('--------- Backbone with predefined descriptor Scale ---------')
    test_losses['Backbone_S'] = trainer.cross_validation(dataset_backbone, model_backbone_S, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mol_collate_gcn.collate_egcn_scale)
    print('test loss (Backbone_S): ' + str(test_losses['Backbone_S']))

    print('--------- Backbone with predefined descriptors ---------')
    test_losses['Backbone'] = trainer.cross_validation(dataset_backbone, model_backbone, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mol_collate_gcn.collate_egcn)
    print('test loss (Backbone): ' + str(test_losses['Backbone']))


    #------------------------ concatenation + descriptor selection ------------------------#
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
    test_losses['Backbone_concat'] = trainer.cross_validation(dataset, model_concat_ds, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, descriptors)
    print('test loss (Backbone_concat): ' + str(test_losses['Backbone_concat']))


    # #------------------------ kronecker-product + descriptor selection ------------------------#
    # print('--------- kronecker-product with 3 descriptors ---------')
    # test_losses['kronecker_3'] = trainer.cross_validation(dataset, model_kronecker_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_3)
    # print('test loss (kronecker_3): ' + str(test_losses['kronecker_3']))

    # print('--------- kronecker-product with 5 descriptors ---------')
    # test_losses['kronecker_5'] = trainer.cross_validation(dataset, model_kronecker_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_5)
    # print('test loss (kronecker_5): ' + str(test_losses['kronecker_5']))

    # print('--------- kronecker-product with 7 descriptors ---------')
    # test_losses['kronecker_7'] = trainer.cross_validation(dataset, model_kronecker_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_7)
    # print('test loss (kronecker_7): ' + str(test_losses['kronecker_7']))

    # print('--------- kronecker-product with 10 descriptors ---------')
    # test_losses['kronecker_10'] = trainer.cross_validation(dataset, model_kronecker_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_10)
    # print('test loss (kronecker_10): ' + str(test_losses['kronecker_10']))

    # print('--------- kronecker-product with 20 descriptors ---------')
    # test_losses['kronecker_20'] = trainer.cross_validation(dataset, model_kronecker_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.descriptor_selection_20)
    # print('test loss (kronecker_20): ' + str(test_losses['kronecker_20']))

    # print('--------- kronecker-product with descriptor selection ---------')
    # test_losses['Backbone_Fusion'] = trainer.cross_validation(dataset, model_Fusion, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, descriptors)
    # print('test loss (Backbone_Fusion): ' + str(test_losses['Backbone_Fusion']))


    print('test_losse:', test_losses)
    print(backbone, DATASET_NAME)

if __name__ == '__main__':
    main()