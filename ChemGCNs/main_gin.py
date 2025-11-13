import random
import torch
import torch.nn as nn

import utils.mol_conv as mc
from utils import trainer
from utils import mol_collate_gcn as mcol
from utils.mol_props import dim_atomic_feat

from model import GCN, GAT, EGCN, GIN
from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K

def main():
    SET_SEED()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataset = mc.read_dataset(DATASET_PATH + '.csv')
    
    random.shuffle(dataset)

    # baselines
    model_GCN = GCN.Net(dim_atomic_feat, 1).to(device)
    model_GAT = GAT.Net(dim_atomic_feat, 1, 4).to(device)
    model_EGCN_R = EGCN.Net(dim_atomic_feat, 1, 1).to(device)
    model_EGCN_S = EGCN.Net(dim_atomic_feat, 1, 2).to(device)
    model_EGCN = EGCN.Net(dim_atomic_feat, 1, 3).to(device)
    model_GIN = GIN.Net(dim_atomic_feat, 1).to(device)

    # loss function
    # criterion = nn.L1Loss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()
    
    # ------------------------ baselines ------------------------#
    # print('--------- GCN ---------')
    # test_losses['GCN'] = trainer.cross_validation(dataset, model_GCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_gcn, trainer.test_gcn, mcol.collate_gcn)
    # print('test loss (GCN): ' + str(test_losses['GCN']))

    # print('--------- GAT ---------')
    # test_losses['GAT'] = trainer.cross_validation(dataset, model_GAT, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_gcn, trainer.test_gcn, mcol.collate_gcn)
    # print('test loss (GAT): ' + str(test_losses['GAT']))

    # print('--------- EGCN_RING ---------')
    # test_losses['EGCN_R'] = trainer.cross_validation(dataset, model_EGCN_R, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.collate_egcn_ring)
    # print('test loss (EGCN_RING): ' + str(test_losses['EGCN_R']))

    # print('--------- EGCN_SCALE ---------')
    # test_losses['EGCN_S'] = trainer.cross_validation(dataset, model_EGCN_S, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.collate_egcn_scale)
    # print('test loss (EGCN_SCALE): ' + str(test_losses['EGCN_S']))

    # print('--------- EGCN ---------')
    # test_losses['EGCN'] = trainer.cross_validation(dataset, model_EGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, mcol.collate_egcn)
    # print('test loss (EGCN): ' + str(test_losses['EGCN']))

    print('--------- GIN ---------')
    test_losses['GIN'] = trainer.cross_validation(dataset, model_GIN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_gcn, trainer.test_gcn, mcol.collate_gcn)
    print('test loss (GIN): ' + str(test_losses['GIN']))

    print('test_losse:', test_losses)
    print(DATASET_NAME)

if __name__ == '__main__':
    main()