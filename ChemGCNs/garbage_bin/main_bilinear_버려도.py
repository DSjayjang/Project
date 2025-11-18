import copy
import random
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils.mol_conv as mc
from utils import mol_collate, evaluation_bilinear
from utils.utils import weight_reset
from utils.mol_props import dim_atomic_feat

from model import KROVEX, Bilinear_Attn, Bilinear_Form # bilinear 지우기
from configs import config
from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K

def main():
    SET_SEED()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if DATASET_NAME == 'freesolv':
        print('DATASET_NAME: ', DATASET_NAME)
        dataset = mc.read_dataset_freesolv(DATASET_PATH + '.csv')
        from utils.ablation import mol_collate_freesolv as mcol
        num_descriptors = 50
        descriptors = mol_collate.descriptor_selection_freesolv

    elif DATASET_NAME == 'esol':
        print('DATASET_NAME: ', DATASET_NAME)
        dataset = mc.read_dataset_esol(DATASET_PATH + '.csv')
        from utils.ablation import mol_collate_esol as mcol
        num_descriptors = 63
        descriptors = mol_collate.descriptor_selection_esol

    elif DATASET_NAME == 'lipo':
        print('DATASET_NAME: ', DATASET_NAME)
        dataset = mc.read_dataset_lipo(DATASET_PATH + '.csv')
        from utils.ablation import mol_collate_lipo as mcol
        num_descriptors = 25
        descriptors = mol_collate.descriptor_selection_lipo

    elif DATASET_NAME == 'scgas':
        print('DATASET_NAME: ', DATASET_NAME)
        global BATCH_SIZE
        BATCH_SIZE = 128
        dataset = mc.read_dataset_scgas(DATASET_PATH + '.csv')
        from utils.ablation import mol_collate_scgas as mcol
        num_descriptors = 23
        descriptors = mol_collate.descriptor_selection_scgas

    random.shuffle(dataset)
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = config.SEED)

    # kronecker-product + descriptor selection
    model_bilinear_attn = Bilinear_Attn.Net(dim_atomic_feat, 1, 3).to(device)

    # loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    val_losses = dict()

    # evaluation
    print('kronecker-product fusion with descriptor selection')
    val_losses['bilinear_attn'], best_model, best_k = evaluation_bilinear.cross_validation(train_dataset, model_bilinear_attn, criterion, K, BATCH_SIZE, MAX_EPOCHS, evaluation_bilinear.train_model, evaluation_bilinear.val_model, mcol.descriptor_selection_3)
    print('Val loss (bilinear_attnX): ' + str(val_losses['bilinear_attn']))

    final_model = copy.deepcopy(best_model)
    final_model.apply(weight_reset) # initializing weights

    optimizer = optim.Adam(final_model.parameters(), weight_decay=0.01)

    # 전체 트레이닝용 dataset
    train_data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = mcol.descriptor_selection_3)
    final_train_loss = evaluation_bilinear.train_model(final_model, criterion, optimizer, train_data_loader, MAX_EPOCHS)

    # 트레이닝 평가용
    evaluation_bilinear.collect_train_preds(final_model, criterion, train_data_loader)

    # final test
    test_data_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = mcol.descriptor_selection_3)
    test_loss, final_preds = evaluation_bilinear.test_model(final_model, criterion, test_data_loader)

    print('best_k-fold:', best_k)
    print('after k-fold, averaging of val_losses:', val_losses)
    print('test_losse:', test_loss)

    print(DATASET_NAME, criterion)
    
if __name__ == '__main__':
    main()