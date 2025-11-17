import copy
import random
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils.mol_conv as mc
import utils.mol_conv_scaffold_old as mcsf
from utils import mol_collate_gcn as mcol

from utils import mol_collate, evaluation
from utils.utils import weight_reset
from utils.mol_props import dim_atomic_feat

from model import GCN, EGCN
from configs import config
from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K

import deepchem as dc

def main():
    SET_SEED()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if DATASET_NAME == 'freesolv':
        print('DATASET_NAME: ', DATASET_NAME)
        X, y, ids = mcsf.read_dataset_scaffold(DATASET_PATH + '.csv')
        dataset = dc.data.DiskDataset.from_numpy(X=X, y=y, ids=ids)
        samples = mc.read_dataset(DATASET_PATH + '.csv')

    elif DATASET_NAME == 'esol':
        print('DATASET_NAME: ', DATASET_NAME)
        X, y, ids = mcsf.read_dataset_scaffold(DATASET_PATH + '.csv')
        dataset = dc.data.DiskDataset.from_numpy(X=X, y=y, ids=ids)
        samples = mc.read_dataset(DATASET_PATH + '.csv')

    elif DATASET_NAME == 'lipo':
        print('DATASET_NAME: ', DATASET_NAME)
        dataset = mc.read_dataset_lipo(DATASET_PATH + '.csv')

    elif DATASET_NAME == 'scgas':
        print('DATASET_NAME: ', DATASET_NAME)
        global BATCH_SIZE
        BATCH_SIZE = 128
        X, y, ids = mcsf.read_dataset_scaffold(DATASET_PATH + '.csv')
        dataset = dc.data.DiskDataset.from_numpy(X=X, y=y, ids=ids)
        samples = mc.read_dataset(DATASET_PATH + '.csv')

    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    train_idx, _, test_idx = scaffoldsplitter.split(dataset)

    train_dataset = [samples[i] for i in train_idx]
    test_dataset = [samples[i] for i in test_idx]

    # kronecker-product + descriptor selection
    model_list = [
    ('GCN', GCN.Net(dim_atomic_feat, 1).to(device), mcol.collate_gcn),
    ('EGCN_R', EGCN.Net(dim_atomic_feat, 1, 1).to(device), mcol.collate_egcn_ring),
    ('EGCN_S', EGCN.Net(dim_atomic_feat, 1, 2).to(device), mcol.collate_egcn_scale),
    ('EGCN', EGCN.Net(dim_atomic_feat, 1, 3).to(device), mcol.collate_egcn),
    ]

    # loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    val_losses = dict()

    val_loss_list = []
    test_loss_list = []
    model_name_list = []

    for model_name, model_instance, desc in model_list:
        if model_name == 'GCN':
            val_losses[model_name], best_model, best_k = evaluation.cross_validation(
                train_dataset, model_instance, criterion, K, BATCH_SIZE, MAX_EPOCHS, evaluation.train_model_gcn, evaluation.val_model_gcn, desc)

            # evaluation
            print(f'[{model_name}] Val loss: {val_losses[model_name]}')

            final_model = copy.deepcopy(best_model)
            final_model.apply(weight_reset) # initializing weights

            optimizer = optim.Adam(final_model.parameters(), weight_decay=0.01)

            # 전체 트레이닝용 dataset
            train_data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = desc)
            final_train_loss = evaluation.train_model_gcn(final_model, criterion, optimizer, train_data_loader, MAX_EPOCHS)

            # 트레이닝 평가용
            evaluation.collect_train_preds_gcn(final_model, criterion, train_data_loader)

            # final test
            test_data_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = desc)
            test_loss, final_preds = evaluation.test_model_gcn(final_model, criterion, test_data_loader)

            val_loss_list.append(val_losses[model_name])
            test_loss_list.append(test_loss)
            model_name_list.append(model_name)

            print(f"[{model_name}] best_k-fold: {best_k}")
            print(f"[{model_name}] final test_loss: {test_loss}\n")

        else:
            val_losses[model_name], best_model, best_k = evaluation.cross_validation(
            train_dataset, model_instance, criterion, K, BATCH_SIZE, MAX_EPOCHS, evaluation.train_model, evaluation.val_model, desc)

            # evaluation
            print(f'[{model_name}] Val loss: {val_losses[model_name]}')

            final_model = copy.deepcopy(best_model)
            final_model.apply(weight_reset) # initializing weights

            optimizer = optim.Adam(final_model.parameters(), weight_decay=0.01)

            # 전체 트레이닝용 dataset
            train_data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = desc)
            final_train_loss = evaluation.train_model(final_model, criterion, optimizer, train_data_loader, MAX_EPOCHS)

            # 트레이닝 평가용
            evaluation.collect_train_preds(final_model, criterion, train_data_loader)

            # final test
            test_data_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = desc)
            test_loss, final_preds = evaluation.test_model(final_model, criterion, test_data_loader)

            val_loss_list.append(val_losses[model_name])
            test_loss_list.append(test_loss)
            model_name_list.append(model_name)

            print(f"[{model_name}] best_k-fold: {best_k}")
            print(f"[{model_name}] final test_loss: {test_loss}\n")

    print("===== Summary of All Val Losses =====")
    for name, loss in zip(model_name_list, val_loss_list):
        print(f"{name} : {loss}")

    print("===== Summary of All Final Test Losses =====")
    for name, loss in zip(model_name_list, test_loss_list):
        print(f"{name} : {loss}")

    print(DATASET_NAME, criterion)
    
if __name__ == '__main__':
    main()
