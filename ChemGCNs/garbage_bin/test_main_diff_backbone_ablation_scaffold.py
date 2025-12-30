import random
import torch
import torch.nn as nn

import utils.mol_conv_scaffold as mc
from utils import trainer_scaffold
from utils import mol_collate
from utils.mol_props import dim_atomic_feat


from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K, SEED

backbone = 'GCN' # [GCN, GAT, GIN, SAGE]

def main():
    SET_SEED()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if DATASET_NAME == 'freesolv':
        print('DATASET_NAME: ', DATASET_NAME)
        from utils.ablation import mol_collate_freesolv as mcol
        dataset, smiles_list = mc.read_dataset_freesolv(DATASET_PATH + '.csv')
        num_descriptors = 50
        descriptors = mol_collate.descriptor_selection_freesolv

    elif DATASET_NAME == 'esol':
        print('DATASET_NAME: ', DATASET_NAME)
        from utils.ablation import mol_collate_esol as mcol
        dataset, smiles_list = mc.read_dataset_esol(DATASET_PATH + '.csv')
        num_descriptors = 63
        descriptors = mol_collate.descriptor_selection_esol

    elif DATASET_NAME == 'lipo':
        print('DATASET_NAME: ', DATASET_NAME)
        from utils.ablation import mol_collate_lipo as mcol
        dataset, smiles_list = mc.read_dataset_lipo(DATASET_PATH + '.csv')
        num_descriptors = 25
        descriptors = mol_collate.descriptor_selection_lipo

    elif DATASET_NAME == 'scgas':
        print('DATASET_NAME: ', DATASET_NAME)
        global BATCH_SIZE
        BATCH_SIZE = 256
        from utils.ablation import mol_collate_scgas as mcol
        dataset, smiles_list = mc.read_dataset_scgas(DATASET_PATH + '.csv')
        num_descriptors = 23
        descriptors = mol_collate.descriptor_selection_scgas

    elif DATASET_NAME == 'solubility_only3':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 512
        from utils.ablation import mol_collate_solubility as mcol
        dataset, smiles_list = mc.read_dataset_solubility(DATASET_PATH + '.csv')
        num_descriptors = 30
        descriptors = mol_collate.descriptor_selection_solubility

    folds = mc.scaffold_kfold_split(smiles_list, K)

    if backbone == 'GCN':
        from model import GCN
        from utils import mol_collate_vanilla
        dataset_backbone, smiles_list_backbone = mc.read_dataset(DATASET_PATH + '.csv')

        model_backbone = GCN.Net(dim_atomic_feat, 1).to(device)
        
        # EGCN
        from model import EGCN
        model_backbone_R = EGCN.Net(dim_atomic_feat, 1, 1).to(device)
        model_backbone_S = EGCN.Net(dim_atomic_feat, 1, 2).to(device)
        model_backbone_E = EGCN.Net(dim_atomic_feat, 1, 3).to(device)

        # GCN + concatenation + descriptor selection
        from model import EGCN_DS
        model_concat_3 = EGCN_DS.concat_Net_3(dim_atomic_feat, 1, 3).to(device)
        model_concat_5 = EGCN_DS.concat_Net_5(dim_atomic_feat, 1, 5).to(device)
        model_concat_7 = EGCN_DS.concat_Net_7(dim_atomic_feat, 1, 7).to(device)
        model_concat_10 = EGCN_DS.concat_Net_10(dim_atomic_feat, 1, 10).to(device)
        model_concat_20 = EGCN_DS.concat_Net_20(dim_atomic_feat, 1, 20).to(device)
        model_concat_ds = EGCN_DS.concat_Net(dim_atomic_feat, 1, num_descriptors).to(device)

        # GCN + kronecker-product + descriptor selection
        from model import KROVEX
        model_kronecker_3 = KROVEX.kronecker_Net_3(dim_atomic_feat, 1, 3).to(device)
        model_kronecker_5 = KROVEX.kronecker_Net_5(dim_atomic_feat, 1, 5).to(device)
        model_kronecker_7 = KROVEX.kronecker_Net_7(dim_atomic_feat, 1, 7).to(device)
        model_kronecker_10 = KROVEX.kronecker_Net_10(dim_atomic_feat, 1, 10).to(device)
        model_kronecker_20 = KROVEX.kronecker_Net_20(dim_atomic_feat, 1, 20).to(device)
        model_Fusion = KROVEX.Net(dim_atomic_feat, 1, num_descriptors).to(device)

    elif backbone == 'GAT':
        from model import GAT
        from utils import mol_collate_vanilla
        dataset_backbone, smiles_list_backbone = mc.read_dataset(DATASET_PATH + '.csv')

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
    
    elif backbone == 'GIN':
        from model import GIN
        from utils import mol_collate_vanilla
        dataset_backbone, smiles_list_backbone = mc.read_dataset(DATASET_PATH + '.csv')

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
        print('아직 모델 정의 안됨')

    folds_backbone = mc.scaffold_kfold_split(smiles_list_backbone, K)

    # loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()

    print(f'{backbone}, {DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')

    #------------------------ Backbone ------------------------#
    print('--------- Vanilla Backbone ---------')
    test_losses['Backbone'] = trainer_scaffold.cross_validation(dataset_backbone, model_backbone, criterion, folds_backbone, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_gcn, trainer_scaffold.test_gcn, mol_collate_vanilla.collate_gcn)
    print('test loss (Backbone): ' + str(test_losses['Backbone']))

    print('--------- Backbone with predefined descriptor Ring ---------')
    test_losses['Backbone_R'] = trainer_scaffold.cross_validation(dataset_backbone, model_backbone_R, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mol_collate_vanilla.collate_egcn_ring)
    print('test loss (Backbone_R): ' + str(test_losses['Backbone_R']))

    print('--------- Backbone with predefined descriptor Scale ---------')
    test_losses['Backbone_S'] = trainer_scaffold.cross_validation(dataset_backbone, model_backbone_S, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mol_collate_vanilla.collate_egcn_scale)
    print('test loss (Backbone_S): ' + str(test_losses['Backbone_S']))

    print('--------- Backbone with predefined descriptors ---------')
    test_losses['Backbone_E'] = trainer_scaffold.cross_validation(dataset_backbone, model_backbone_E, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mol_collate_vanilla.collate_egcn)
    print('test loss (Backbone_E): ' + str(test_losses['Backbone_E']))


    #------------------------ concatenation + descriptor selection ------------------------#
    print('--------- concatenation with 3 descriptors ---------')
    test_losses['concat_3'] = trainer_scaffold.cross_validation(dataset, model_concat_3, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_3)
    print('test loss (concat_3): ' + str(test_losses['concat_3']))

    print('--------- concatenation with 5 descriptors ---------')
    test_losses['concat_5'] = trainer_scaffold.cross_validation(dataset, model_concat_5, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_5)
    print('test loss (concat_5): ' + str(test_losses['concat_5']))

    print('--------- concatenation with 7 descriptors ---------')
    test_losses['concat_7'] = trainer_scaffold.cross_validation(dataset, model_concat_7, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_7)
    print('test loss (concat_7): ' + str(test_losses['concat_7']))

    print('--------- concatenation with 10 descriptors ---------')
    test_losses['concat_10'] = trainer_scaffold.cross_validation(dataset, model_concat_10, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_10)
    print('test loss (concat_10): ' + str(test_losses['concat_10']))

    print('--------- concatenation with 20 descriptors ---------')
    test_losses['concat_20'] = trainer_scaffold.cross_validation(dataset, model_concat_20, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_20)
    print('test loss (concat_20): ' + str(test_losses['concat_20']))

    print('--------- concatenation with descriptor selection ---------')
    test_losses['Backbone_concat'] = trainer_scaffold.cross_validation(dataset, model_concat_ds, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, descriptors)
    print('test loss (Backbone_concat): ' + str(test_losses['Backbone_concat']))


    #------------------------ kronecker-product + descriptor selection ------------------------#
    print('--------- kronecker-product with 3 descriptors ---------')
    test_losses['kronecker_3'] = trainer_scaffold.cross_validation(dataset, model_kronecker_3, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_3)
    print('test loss (kronecker_3): ' + str(test_losses['kronecker_3']))

    print('--------- kronecker-product with 5 descriptors ---------')
    test_losses['kronecker_5'] = trainer_scaffold.cross_validation(dataset, model_kronecker_5, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_5)
    print('test loss (kronecker_5): ' + str(test_losses['kronecker_5']))

    print('--------- kronecker-product with 7 descriptors ---------')
    test_losses['kronecker_7'] = trainer_scaffold.cross_validation(dataset, model_kronecker_7, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_7)
    print('test loss (kronecker_7): ' + str(test_losses['kronecker_7']))

    print('--------- kronecker-product with 10 descriptors ---------')
    test_losses['kronecker_10'] = trainer_scaffold.cross_validation(dataset, model_kronecker_10, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_10)
    print('test loss (kronecker_10): ' + str(test_losses['kronecker_10']))

    print('--------- kronecker-product with 20 descriptors ---------')
    test_losses['kronecker_20'] = trainer_scaffold.cross_validation(dataset, model_kronecker_20, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.descriptor_selection_20)
    print('test loss (kronecker_20): ' + str(test_losses['kronecker_20']))

    print('--------- kronecker-product with descriptor selection ---------')
    test_losses['Backbone_Fusion'] = trainer_scaffold.cross_validation(dataset, model_Fusion, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, descriptors)
    print('test loss (Backbone_Fusion): ' + str(test_losses['Backbone_Fusion']))


    print('test_losse:', test_losses)
    print(f'{backbone}, {DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')


if __name__ == '__main__':
    main()