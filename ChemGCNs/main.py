import random
import torch
import torch.nn as nn

from utils import trainer
from utils.mol_props import dim_atomic_feat
from utils.mol_collate import collate_kfgcn_freesolv
from configs.config import SET_SEED, DATASET, BATCH_SIZE, MAX_EPOCHS, K
from utils.mol_dataset import MoleculeDataset

from model import KFGCN

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
    model_KFGCN = KFGCN.Net(dim_atomic_feat, 1, 37).to(device)

    # define loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    # train and evaluate competitors
    test_losses = dict()

    print('--------- KFGCN ---------')
    test_losses['KFGCN'] = trainer.cross_validation(dataset, model_KFGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv)
    print('test loss (KFGCN): ' + str(test_losses['KFGCN']))

    print(test_losses)


if __name__ == '__main__':
    main()