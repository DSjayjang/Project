import copy
import random
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.mol_props import dim_atomic_feat

import utils.mol_conv as mc
from utils import mol_collate, evaluation
from utils.utils import weight_reset

from model import KDGCN
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

    # kronecker-product + descriptor selection
    model_KDGCN = KDGCN.KDGCN_Net(dim_atomic_feat, 1, 63).to(device)

    # loss
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

   # train and evaluate
    val_losses = dict()

    # evaluation
    print('kronecker-product fusion with descriptor selection')
    val_losses['KDGCN'], best_model, best_k = evaluation.cross_validation(train_dataset, model_KDGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, evaluation.train_model, evaluation.val_model, mol_collate.collate_kfgcn_esol)
    print('Val loss (KDGCN): ' + str(val_losses['KDGCN']))

    final_model = copy.deepcopy(best_model)
    final_model.apply(weight_reset) # initializing weights


    # optimizer = optim.Adam(final_model.parameters(), weight_decay=0.01)

    # 전체 트레이닝용 dataset
    train_data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = mol_collate.collate_kfgcn_esol)

    # 트레이닝 평가용
    evaluation.collect_train_preds(final_model, criterion, train_data_loader)

    # final test
    test_data_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = mol_collate.collate_kfgcn_esol)
    test_loss, final_preds = evaluation.test_model(final_model, criterion, test_data_loader)

    print('best_k-fold:', best_k)
    print('after k-fold, averaging of val_losses:', val_losses)
    print('test_losse:', test_loss)

if __name__ == '__main__':
    main()