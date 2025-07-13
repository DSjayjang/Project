from model import KFGCN
from utils import trainer
import utils.mol_conv_esol as mc
from utils import mol_props
from utils.mol_collate import collate_kfgcn_esol
from utils.test_mol.mol_collate_esol import collate_kfgcn_esol_3, collate_kfgcn_esol_5, collate_kfgcn_esol_7, collate_kfgcn_esol_10, collate_kfgcn_esol_20
from utils.mol_props import dim_atomic_feat
from configs.config import SET_SEED, DATASET, BATCH_SIZE, MAX_EPOCHS, K

from utils.mol_dataset import MoleculeDataset_esol

import torch
import torch.nn as nn
import random


# 재현성-난수 고정
def main():
    SET_SEED()

    # check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load train, validation, and test datasets
    print('Data loading...')
    dataset = mc.read_dataset('datasets/esol.csv')
    # dataset = MoleculeDataset_esol(DATASET).data
    random.shuffle(dataset)

    model_KFGCN = KFGCN.Net(mol_props.dim_atomic_feat, 1, mc.dim_self_feat).to(device)
    model_KFGCN3 = KFGCN.Net(dim_atomic_feat, 1, 3).to(device)
    model_KFGCN5 = KFGCN.Net(dim_atomic_feat, 1, 5).to(device)
    model_KFGCN7 = KFGCN.Net(dim_atomic_feat, 1, 7).to(device)
    model_KFGCN10 = KFGCN.Net(dim_atomic_feat, 1, 10).to(device)
    model_KFGCN20 = KFGCN.Net(dim_atomic_feat, 1, 20).to(device)

    # define loss function
    criterion = nn.L1Loss(reduction='sum') # MAE
    # criterion = nn.MSELoss(reduction='sum') # MSE

    # train and evaluate competitors
    test_losses = dict()

    #------------------------ Self Feature ------------------------#

    print('--------- KFGCN ---------')
    test_losses['KFGCN3'] = trainer.cross_validation(dataset, model_KFGCN3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_3)
    print('test loss (KFGCN3): ' + str(test_losses['KFGCN3']))

    test_losses['KFGCN5'] = trainer.cross_validation(dataset, model_KFGCN5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_5)
    print('test loss (KFGCN5): ' + str(test_losses['KFGCN5']))

    test_losses['KFGCN7'] = trainer.cross_validation(dataset, model_KFGCN7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_7)
    print('test loss (KFGCN7): ' + str(test_losses['KFGCN7']))

    test_losses['KFGCN10'] = trainer.cross_validation(dataset, model_KFGCN10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_10)
    print('test loss (KFGCN10): ' + str(test_losses['KFGCN10']))

    test_losses['KFGCN20'] = trainer.cross_validation(dataset, model_KFGCN20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_20)
    print('test loss (KFGCN20): ' + str(test_losses['KFGCN20']))

    # print('--------- Outer EGCN_elastic ---------')
    # test_losses['Outer_EGCN_elastic'] = trainer.cross_validation(dataset, model_KFGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol)
    # print('test loss (Outer_EGCN_elastic): ' + str(test_losses['Outer_EGCN_elastic']))

    print(test_losses)

if __name__ == '__main__':
    main()