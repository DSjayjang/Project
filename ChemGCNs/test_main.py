import random
import torch
import torch.nn as nn

from utils import trainer
from utils.mol_props import dim_atomic_feat
from utils.test_mol.mol_collate_gcn import collate_gcn, collate_egcn_ring, collate_egcn_scale, collate_egcn
from utils.test_mol.mol_collate_esol import collate_kfgcn_esol_3, collate_kfgcn_esol_5, collate_kfgcn_esol_7, collate_kfgcn_esol_10, collate_kfgcn_esol_20
from configs.config import SET_SEED, DATASET, BATCH_SIZE, MAX_EPOCHS, K
from utils.test_mol.mol_dataset_gcn import MoleculeDataset
from utils.mol_dataset import MoleculeDataset_esol

from model.test_model import GCN, EGCN
from model.test_model import CFGCN_3, CFGCN_5, CFGCN_7, CFGCN_10, CFGCN_20
from model.test_model import KFGCN_3, KFGCN_5, KFGCN_7, KFGCN_10, KFGCN_20
from model import KFGCN

def main():
    # 시드 고정
    SET_SEED()

    # check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Data Load
    dataset = MoleculeDataset(DATASET).data
    # dataset = MoleculeDataset_esol(DATASET).data
    random.shuffle(dataset)

    # Model
    model_GCN = GCN.Net(dim_atomic_feat, 1).to(device)
    model_EGCN_R = EGCN.Net(dim_atomic_feat, 1, 1).to(device)
    model_EGCN_S = EGCN.Net(dim_atomic_feat, 1, 2).to(device)
    model_EGCN = EGCN.Net(dim_atomic_feat, 1, 3).to(device)

    # model_CFGCN3 = CFGCN_3.Net(dim_atomic_feat, 1, 3).to(device)
    # model_CFGCN5 = CFGCN_5.Net(dim_atomic_feat, 1, 5).to(device)
    # model_CFGCN7 = CFGCN_7.Net(dim_atomic_feat, 1, 7).to(device)
    # model_CFGCN10 = CFGCN_10.Net(dim_atomic_feat, 1, 10).to(device)
    # model_CFGCN20 = CFGCN_20.Net(dim_atomic_feat, 1, 20).to(device)

    # model_KFGCN3 = KFGCN_3.Net(dim_atomic_feat, 1, 3).to(device)
    # model_KFGCN5 = KFGCN_5.Net(dim_atomic_feat, 1, 5).to(device)
    # model_KFGCN7 = KFGCN_7.Net(dim_atomic_feat, 1, 7).to(device)
    # model_KFGCN10 = KFGCN_10.Net(dim_atomic_feat, 1, 10).to(device)
    # model_KFGCN20 = KFGCN_20.Net(dim_atomic_feat, 1, 20).to(device)

    # define loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    # train and evaluate competitors
    test_losses = dict()
    print('--------- GCN ---------')
    test_losses['GCN'] = trainer.cross_validation(dataset, model_GCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train, trainer.test, collate_gcn)
    print('test loss (GCN): ' + str(test_losses['GCN']))

    test_losses['EGCN_R'] = trainer.cross_validation(dataset, model_EGCN_R, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_egcn_ring)
    print('test loss (EGCN_R): ' + str(test_losses['EGCN_R']))

    test_losses['EGCN_S'] = trainer.cross_validation(dataset, model_EGCN_S, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_egcn_scale)
    print('test loss (EGCN_S): ' + str(test_losses['EGCN_S']))

    test_losses['EGCN'] = trainer.cross_validation(dataset, model_EGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_egcn)
    print('test loss (EGCN): ' + str(test_losses['EGCN']))


    # print('--------- CFGCN ---------')
    # test_losses['CFGCN3'] = trainer.cross_validation(dataset, model_CFGCN3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_3)
    # print('test loss (CFGCN3): ' + str(test_losses['CFGCN3']))

    # test_losses['CFGCN5'] = trainer.cross_validation(dataset, model_CFGCN5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_5)
    # print('test loss (CFGCN5): ' + str(test_losses['CFGCN5']))

    # test_losses['CFGCN7'] = trainer.cross_validation(dataset, model_CFGCN7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_7)
    # print('test loss (CFGCN7): ' + str(test_losses['CFGCN7']))

    # test_losses['CFGCN10'] = trainer.cross_validation(dataset, model_CFGCN10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_10)
    # print('test loss (CFGCN10): ' + str(test_losses['CFGCN10']))

    # test_losses['CFGCN20'] = trainer.cross_validation(dataset, model_CFGCN20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_20)
    # print('test loss (CFGCN20): ' + str(test_losses['CFGCN20']))


    # print('--------- KFGCN ---------')
    # test_losses['KFGCN3'] = trainer.cross_validation(dataset, model_KFGCN3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_3)
    # print('test loss (KFGCN3): ' + str(test_losses['KFGCN3']))

    # test_losses['KFGCN5'] = trainer.cross_validation(dataset, model_KFGCN5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_5)
    # print('test loss (KFGCN5): ' + str(test_losses['KFGCN5']))

    # test_losses['KFGCN7'] = trainer.cross_validation(dataset, model_KFGCN7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_7)
    # print('test loss (KFGCN7): ' + str(test_losses['KFGCN7']))

    # test_losses['KFGCN10'] = trainer.cross_validation(dataset, model_KFGCN10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_10)
    # print('test loss (KFGCN10): ' + str(test_losses['KFGCN10']))

    # test_losses['KFGCN20'] = trainer.cross_validation(dataset, model_KFGCN20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_20)
    # print('test loss (KFGCN20): ' + str(test_losses['KFGCN20']))

    print(test_losses)


if __name__ == '__main__':
    main()