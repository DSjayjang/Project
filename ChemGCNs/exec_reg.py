import random
import torch
import torch.nn as nn

# import utils.mol_dataset as mc
from utils import trainer
from utils.test_mol import mol_collate_gcn as mcol

from model.test_model import GCN
from model.test_model import GAT
from model.test_model import EGCN
from configs.config import SET_SEED, DATASET, BATCH_SIZE, MAX_EPOCHS, K

from utils.mol_dataset import MoleculeDataset_esol

def main():
    # seed
    SET_SEED()
    
    # check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load train, validation, and test datasets
    print('Data loading...')
    dataset = MoleculeDataset_esol(DATASET).data
    random.shuffle(dataset)

    model_GCN = GCN.Net(mc.dim_atomic_feat, 1).to(device)
    model_GAT = GAT.Net(mc.dim_atomic_feat, 1, 4).to(device)
    model_EGCN_R = EGCN.Net(mc.dim_atomic_feat, 1, 1).to(device)
    model_EGCN_S = EGCN.Net(mc.dim_atomic_feat, 1, 2).to(device)
    model_EGCN = EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)

    # loss
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()

    print('--------- Outer EGCN_elastic ---------')
    test_losses['Outer_EGCN_elastic'] = trainer.cross_validation(dataset, model_Outer_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic)
    print('test loss (Outer_EGCN_elastic): ' + str(test_losses['Outer_EGCN_elastic']))
    print('test_losse:', test_losses)

if __name__ == '__main__':
    main()