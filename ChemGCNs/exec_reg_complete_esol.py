import random
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from utils.mol_props import dim_atomic_feat

import utils.mol_conv as mc
from utils import trainer
from utils import mol_collate
from utils.test_mol import mol_collate_esol as mcol

from model import CDGCN, KDGCN
from configs import config
from configs.config import SET_SEED, DATASET, BATCH_SIZE, MAX_EPOCHS, K

def main():
    # seed
    SET_SEED()
    
    # check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load train, validation, and test datasets
    print('Data loading...')
    dataset = mc.read_dataset_esol(DATASET + '.csv')
    random.shuffle(dataset)
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = config.SEED)


    # concatenation + descriptor selection
    model_CDGCN_3 = CDGCN.CDGCN_Net_3(dim_atomic_feat, 1, 3).to(device)
    model_CDGCN_5 = CDGCN.CDGCN_Net_5(dim_atomic_feat, 1, 5).to(device)
    model_CDGCN_7 = CDGCN.CDGCN_Net_7(dim_atomic_feat, 1, 7).to(device)
    model_CDGCN_10 = CDGCN.CDGCN_Net_10(dim_atomic_feat, 1, 10).to(device)
    model_CDGCN_20 = CDGCN.CDGCN_Net_20(dim_atomic_feat, 1, 20).to(device)

    # kronecker-product + descriptor selection
    model_KDGCN_3 = KDGCN.KDGCN_Net_3(dim_atomic_feat, 1, 3).to(device)
    model_KDGCN_5 = KDGCN.KDGCN_Net_5(dim_atomic_feat, 1, 5).to(device)
    model_KDGCN_7 = KDGCN.KDGCN_Net_7(dim_atomic_feat, 1, 7).to(device)
    model_KDGCN_10 = KDGCN.KDGCN_Net_10(dim_atomic_feat, 1, 10).to(device)
    model_KDGCN_20 = KDGCN.KDGCN_Net_20(dim_atomic_feat, 1, 20).to(device)

    # Self_Feature
    model_CDGCN = CDGCN.CDGCN_Net(dim_atomic_feat, 1, 63).to(device)
    model_KDGCN = KDGCN.KDGCN_Net(dim_atomic_feat, 1, 63).to(device)

    # loss
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

   # train and evaluate competitors
    val_losses = dict()
    test_losses = dict()


    #------------------------ concatenation + descriptor selection ------------------------#

    # feature 3개
    print('--------- EGCN_3 ---------')
    test_losses['EGCN_3'] = trainer.cross_validation(dataset, model_CDGCN_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_3)
    print('test loss (EGCN_3): ' + str(test_losses['EGCN_3']))

    # feature 5개
    print('--------- EGCN_5 ---------')
    test_losses['EGCN_5'] = trainer.cross_validation(dataset, model_CDGCN_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_5)
    print('test loss (EGCN_5): ' + str(test_losses['EGCN_5']))

    # feature 7개
    print('--------- EGCN_7 ---------')
    test_losses['EGCN_7'] = trainer.cross_validation(dataset, model_CDGCN_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_7)
    print('test loss (EGCN_7): ' + str(test_losses['EGCN_7']))

    # feature 10개
    print('--------- EGCN_10 ---------')
    test_losses['EGCN_10'] = trainer.cross_validation(dataset, model_CDGCN_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_10)
    print('test loss (EGCN_10): ' + str(test_losses['EGCN_10']))

    # feature 20개
    print('--------- EGCN_20 ---------')
    test_losses['EGCN_20'] = trainer.cross_validation(dataset, model_CDGCN_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_20)
    print('test loss (EGCN_20): ' + str(test_losses['EGCN_20']))

    # feature 20개
    print('--------- EGCN_elastic ---------')
    test_losses['EGCN_elastic'] = trainer.cross_validation(dataset, model_CDGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mol_collate.collate_kfgcn_esol)
    print('test loss (EGCN_elastic): ' + str(test_losses['EGCN_elastic']))

    #------------------------ kronecker-product + descriptor selection ------------------------#

    # feature 3개
    print('--------- Outer EGCN_3 ---------')
    test_losses['Outer_EGCN_3'] = trainer.cross_validation(dataset, model_KDGCN_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_3)
    print('test loss (Outer_EGCN_3): ' + str(test_losses['Outer_EGCN_3']))

    # feature 5개
    print('--------- Outer EGCN_5 ---------')
    test_losses['Outer_EGCN_5'] = trainer.cross_validation(dataset, model_KDGCN_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_5)
    print('test loss (Outer_EGCN_5): ' + str(test_losses['Outer_EGCN_5']))

    # feature 7개
    print('--------- Outer EGCN_7 ---------')
    test_losses['Outer_EGCN_7'] = trainer.cross_validation(dataset, model_KDGCN_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_7)
    print('test loss (Outer_EGCN_7): ' + str(test_losses['Outer_EGCN_7']))

    # feature 10개
    print('--------- Outer EGCN_10 ---------')
    test_losses['Outer_EGCN_10'] = trainer.cross_validation(dataset, model_KDGCN_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_10)
    print('test loss (Outer_EGCN_10): ' + str(test_losses['Outer_EGCN_10']))

    # feature 20개
    print('--------- Outer EGCN_20 ---------')
    test_losses['Outer_EGCN_20'] = trainer.cross_validation(dataset, model_KDGCN_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mcol.collate_kfgcn_20)
    print('test loss (Outer_EGCN_20): ' + str(test_losses['Outer_EGCN_20']))

    print('--------- Outer EGCN_elastic ---------')
    test_losses['Outer_EGCN_elastic'] = trainer.cross_validation(dataset, model_KDGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, mol_collate.collate_kfgcn_esol)
    print('test loss (Outer_EGCN_elastic): ' + str(test_losses['Outer_EGCN_elastic']))

    print('test_losse:', test_losses)
 
if __name__ == '__main__':
    main()