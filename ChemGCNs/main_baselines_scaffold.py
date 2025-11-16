import random
import torch
import torch.nn as nn

import utils.mol_conv_scaffold as mc
from utils import trainer_scaffold
from utils import mol_collate_gcn as mcol
from utils.mol_props import dim_atomic_feat

from model import GCN, GAT, EGCN
from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K, SEED

def main():
    SET_SEED()
    global BATCH_SIZE
    if DATASET_NAME == 'scgas': 
        BATCH_SIZE = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataset, smiles_list = mc.read_dataset(DATASET_PATH + '.csv')
    
    folds = mc.scaffold_kfold_split(smiles_list, K)

    # baselines
    model_GCN = GCN.Net(dim_atomic_feat, 1).to(device)
    # model_GAT = GAT.Net(dim_atomic_feat, 1, 4).to(device)
    model_EGCN_R = EGCN.Net(dim_atomic_feat, 1, 1).to(device)
    model_EGCN_S = EGCN.Net(dim_atomic_feat, 1, 2).to(device)
    model_EGCN = EGCN.Net(dim_atomic_feat, 1, 3).to(device)

    # loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()

    print(f'{DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')

    # ------------------------ baselines ------------------------#
    print('--------- GCN ---------')
    test_losses['GCN'] = trainer_scaffold.cross_validation(dataset, model_GCN, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_gcn, trainer_scaffold.test_gcn, mcol.collate_gcn)
    print('test loss (GCN): ' + str(test_losses['GCN']))

    # print('--------- GAT ---------')
    # test_losses['GAT'] = trainer.cross_validation(dataset, model_GAT, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_gcn, trainer.test_gcn, mcol.collate_gcn)
    # print('test loss (GAT): ' + str(test_losses['GAT']))

    print('--------- EGCN_RING ---------')
    test_losses['EGCN_R'] = trainer_scaffold.cross_validation(dataset, model_EGCN_R, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.collate_egcn_ring)
    print('test loss (EGCN_RING): ' + str(test_losses['EGCN_R']))

    print('--------- EGCN_SCALE ---------')
    test_losses['EGCN_S'] = trainer_scaffold.cross_validation(dataset, model_EGCN_S, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.collate_egcn_scale)
    print('test loss (EGCN_SCALE): ' + str(test_losses['EGCN_S']))

    print('--------- EGCN ---------')
    test_losses['EGCN'] = trainer_scaffold.cross_validation(dataset, model_EGCN, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer_scaffold.train_model, trainer_scaffold.test_model, mcol.collate_egcn)
    print('test loss (EGCN): ' + str(test_losses['EGCN']))
    
    print('test_losse:', test_losses)
    print(f'{DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')

if __name__ == '__main__':
    main()