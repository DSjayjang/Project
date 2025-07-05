import random
import torch
import torch.nn as nn

from utils import trainer
from utils.mol_props import dim_atomic_feat
from utils.mol_collate import collate, collate_emodel_ring, collate_emodel_scale, collate_emodel
from configs.config import SET_SEED, DATASET, BATCH_SIZE, MAX_EPOCHS, K
from utils.mol_dataset import MoleculeDataset

from model import GCN
from model import EGCN

def main():
    # 시드 고정
    SET_SEED()

    # check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Data Load
    dataset = MoleculeDataset(DATASET).data
    # dataset = dataset[:3]
    random.shuffle(dataset)

    # Model
    model_GCN = GCN.Net(dim_atomic_feat, 1).to(device)
    model_EGCN_R = EGCN.Net(dim_atomic_feat, 1, 1).to(device)
    model_EGCN_S = EGCN.Net(dim_atomic_feat, 1, 2).to(device)
    model_EGCN = EGCN.Net(dim_atomic_feat, 1, 3).to(device)

    # define loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    # train and evaluate competitors
    test_losses = dict()


    print('--------- GCN ---------')
    test_losses['GCN'] = trainer.cross_validation(dataset, model_GCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train, trainer.test, collate)
    print('test loss (GCN): ' + str(test_losses['GCN']))

    print('--------- EGCN_RING ---------')
    test_losses['EGCN_R'] = trainer.cross_validation(dataset, model_EGCN_R, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_emodel_ring)
    print('test loss (EGCN_RING): ' + str(test_losses['EGCN_R']))

    print('--------- EGCN_SCALE ---------')
    test_losses['EGCN_S'] = trainer.cross_validation(dataset, model_EGCN_S, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_emodel_scale)
    print('test loss (EGCN_SCALE): ' + str(test_losses['EGCN_S']))

    print('--------- EGCN ---------')
    test_losses['EGCN'] = trainer.cross_validation(dataset, model_EGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_emodel)
    print('test loss (EGCN): ' + str(test_losses['EGCN']))

    print(test_losses)


if __name__ == '__main__':
    main()