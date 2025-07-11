import random
import torch
import torch.nn as nn

from utils import trainer
from utils.mol_props import dim_atomic_feat
from utils.test_mol.mol_collate_gcn import collate_gcn, collate_egcn_ring, collate_egcn_scale, collate_egcn
from utils.test_mol.mol_collate_freesolv import collate_kfgcn_freesolv_3, collate_kfgcn_freesolv_5, collate_kfgcn_freesolv_7, collate_kfgcn_freesolv_10, collate_kfgcn_freesolv_20
from utils.test_mol.mol_collate_esol import collate_kfgcn_esol_3, collate_kfgcn_esol_5, collate_kfgcn_esol_7, collate_kfgcn_esol_10, collate_kfgcn_esol_20

from utils.mol_collate import collate_kfgcn_freesolv, collate_kfgcn_esol

from configs.config import SET_SEED, DATASET, BATCH_SIZE, MAX_EPOCHS, K
from utils.test_mol.mol_dataset_gcn import MoleculeDataset
from utils.mol_dataset import MoleculeDataset_freesolv, MoleculeDataset_esol

from model.test_model import GCN
from model.test_model import EGCN
from model.test_model import CFGCN_3, CFGCN_5, CFGCN_7, CFGCN_10, CFGCN_20
from model.test_model import KFGCN_3, KFGCN_5, KFGCN_7, KFGCN_10, KFGCN_20

from model import KFGCN


# 기존꺼
import utils.test_mol.mol_conv_esol as mc

def main():
    # 시드 고정
    SET_SEED()

    # check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Data Load
    dataset = mc.read_dataset('datasets/esol.csv')
    # dataset = MoleculeDataset_esol(DATASET).data
    # dataset = dataset[:3]
    random.shuffle(dataset)

    # Model
    model_GCN = GCN.Net(dim_atomic_feat, 1).to(device)
    model_EGCN_R = EGCN.Net(dim_atomic_feat, 1, 1).to(device)
    model_EGCN_S = EGCN.Net(dim_atomic_feat, 1, 2).to(device)
    model_EGCN = EGCN.Net(dim_atomic_feat, 1, 3).to(device)

    model_CFGCN_3 = CFGCN_3.Net(dim_atomic_feat, 1, 3).to(device)
    model_CFGCN_5 = CFGCN_5.Net(dim_atomic_feat, 1, 5).to(device)
    model_CFGCN_7 = CFGCN_7.Net(dim_atomic_feat, 1, 7).to(device)
    model_CFGCN_10 = CFGCN_10.Net(dim_atomic_feat, 1, 10).to(device)
    model_CFGCN_20 = CFGCN_20.Net(dim_atomic_feat, 1, 20).to(device)

    model_KFGCN_3 = KFGCN_3.Net(dim_atomic_feat, 1, 3).to(device)
    model_KFGCN_5 = KFGCN_5.Net(dim_atomic_feat, 1, 5).to(device)
    model_KFGCN_7 = KFGCN_7.Net(dim_atomic_feat, 1, 7).to(device)
    model_KFGCN_10 = KFGCN_10.Net(dim_atomic_feat, 1, 10).to(device)
    model_KFGCN_20 = KFGCN_20.Net(dim_atomic_feat, 1, 20).to(device)

    model_KFGCN = KFGCN.Net(dim_atomic_feat, 1, 43).to(device)

    # define loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    # train and evaluate competitors
    test_losses = dict()

    # print('--------- GCN ---------')
    # test_losses['GCN'] = trainer.cross_validation(dataset, model_GCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train, trainer.test, collate_gcn)
    # print('test loss (GCN): ' + str(test_losses['GCN']))

    # print('--------- EGCN_RING ---------')
    # test_losses['EGCN_R'] = trainer.cross_validation(dataset, model_EGCN_R, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_egcn_ring)
    # print('test loss (EGCN_RING): ' + str(test_losses['EGCN_R']))

    # print('--------- EGCN_SCALE ---------')
    # test_losses['EGCN_S'] = trainer.cross_validation(dataset, model_EGCN_S, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_egcn_scale)
    # print('test loss (EGCN_SCALE): ' + str(test_losses['EGCN_S']))

    # print('--------- EGCN ---------')
    # test_losses['EGCN'] = trainer.cross_validation(dataset, model_EGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_egcn)
    # print('test loss (EGCN): ' + str(test_losses['EGCN']))

    # freesolv
    # CFGCN / KFGCN
    # print('--------- CFGCN_3 ---------')
    # test_losses['CFGCN_3'] = trainer.cross_validation(dataset, model_CFGCN_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_3)
    # print('test loss (CFGCN_3): ' + str(test_losses['CFGCN_3']))

    # print('--------- CFGCN_5 ---------')
    # test_losses['CFGCN_5'] = trainer.cross_validation(dataset, model_CFGCN_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_5)
    # print('test loss (CFGCN_5): ' + str(test_losses['CFGCN_5']))

    # print('--------- CFGCN_7 ---------')
    # test_losses['CFGCN_7'] = trainer.cross_validation(dataset, model_CFGCN_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_7)
    # print('test loss (CFGCN_7): ' + str(test_losses['CFGCN_7']))

    # print('--------- CFGCN_10 ---------')
    # test_losses['CFGCN_10'] = trainer.cross_validation(dataset, model_CFGCN_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_10)
    # print('test loss (CFGCN_10): ' + str(test_losses['CFGCN_10']))

    # print('--------- CFGCN_20 ---------')
    # test_losses['CFGCN_20'] = trainer.cross_validation(dataset, model_CFGCN_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_20)
    # print('test loss (CFGCN_20): ' + str(test_losses['CFGCN_20']))


    # print('--------- KFGCN_3 ---------')
    # test_losses['KFGCN_3'] = trainer.cross_validation(dataset, model_KFGCN_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_3)
    # print('test loss (KFGCN_3): ' + str(test_losses['KFGCN_3']))

    # print('--------- KFGCN_5 ---------')
    # test_losses['KFGCN_5'] = trainer.cross_validation(dataset, model_KFGCN_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_5)
    # print('test loss (KFGCN_5): ' + str(test_losses['KFGCN_5']))

    # print('--------- KFGCN_7 ---------')
    # test_losses['KFGCN_7'] = trainer.cross_validation(dataset, model_KFGCN_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_7)
    # print('test loss (KFGCN_7): ' + str(test_losses['KFGCN_7']))

    # print('--------- KFGCN_10 ---------')
    # test_losses['KFGCN_10'] = trainer.cross_validation(dataset, model_KFGCN_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_10)
    # print('test loss (KFGCN_10): ' + str(test_losses['KFGCN_10']))

    # print('--------- KFGCN_20 ---------')
    # test_losses['KFGCN_20'] = trainer.cross_validation(dataset, model_KFGCN_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv_20)
    # print('test loss (KFGCN_20): ' + str(test_losses['KFGCN_20']))

    # print('--------- KFGCN ---------')
    # test_losses['KFGCN'] = trainer.cross_validation(dataset, model_KFGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_freesolv)
    # print('test loss (KFGCN): ' + str(test_losses['KFGCN']))

    # freesolv
    # CFGCN / KFGCN
    # print('--------- CFGCN_3 ---------')
    # test_losses['CFGCN_3'] = trainer.cross_validation(dataset, model_CFGCN_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_3)
    # print('test loss (CFGCN_3): ' + str(test_losses['CFGCN_3']))

    # print('--------- CFGCN_5 ---------')
    # test_losses['CFGCN_5'] = trainer.cross_validation(dataset, model_CFGCN_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_5)
    # print('test loss (CFGCN_5): ' + str(test_losses['CFGCN_5']))

    # print('--------- CFGCN_7 ---------')
    # test_losses['CFGCN_7'] = trainer.cross_validation(dataset, model_CFGCN_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_7)
    # print('test loss (CFGCN_7): ' + str(test_losses['CFGCN_7']))

    # print('--------- CFGCN_10 ---------')
    # test_losses['CFGCN_10'] = trainer.cross_validation(dataset, model_CFGCN_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_10)
    # print('test loss (CFGCN_10): ' + str(test_losses['CFGCN_10']))

    # print('--------- CFGCN_20 ---------')
    # test_losses['CFGCN_20'] = trainer.cross_validation(dataset, model_CFGCN_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_20)
    # print('test loss (CFGCN_20): ' + str(test_losses['CFGCN_20']))


    # print('--------- KFGCN_3 ---------')
    # test_losses['KFGCN_3'] = trainer.cross_validation(dataset, model_KFGCN_3, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_3)
    # print('test loss (KFGCN_3): ' + str(test_losses['KFGCN_3']))

    # print('--------- KFGCN_5 ---------')
    # test_losses['KFGCN_5'] = trainer.cross_validation(dataset, model_KFGCN_5, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_5)
    # print('test loss (KFGCN_5): ' + str(test_losses['KFGCN_5']))

    # print('--------- KFGCN_7 ---------')
    # test_losses['KFGCN_7'] = trainer.cross_validation(dataset, model_KFGCN_7, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_7)
    # print('test loss (KFGCN_7): ' + str(test_losses['KFGCN_7']))

    # print('--------- KFGCN_10 ---------')
    # test_losses['KFGCN_10'] = trainer.cross_validation(dataset, model_KFGCN_10, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_10)
    # print('test loss (KFGCN_10): ' + str(test_losses['KFGCN_10']))

    # print('--------- KFGCN_20 ---------')
    # test_losses['KFGCN_20'] = trainer.cross_validation(dataset, model_KFGCN_20, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol_20)
    # print('test loss (KFGCN_20): ' + str(test_losses['KFGCN_20']))

    print('--------- KFGCN ---------')
    test_losses['KFGCN'] = trainer.cross_validation(dataset, model_KFGCN, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_emodel, trainer.test_emodel, collate_kfgcn_esol)
    print('test loss (KFGCN): ' + str(test_losses['KFGCN']))

    print(test_losses)

if __name__ == '__main__':
    main()