import yaml
import argparse

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from utils.utils import SET_SEED, select_loss
from configs.registry import get_dataset_spec, build_collate_fn, bs
from utils.mol_props import dim_atomic_feat
from utils import evaluation, evaluation_LapPE

def get_parser():
    parser = argparse.ArgumentParser(description='Graph Transformer ~~')
    
    parser.add_argument('--config', default='./configs/config.yaml', help='describe??')
    parser.add_argument('--r-home', type=str, help='...')
    parser.add_argument('--dataset-path', default='./datasets/', type=str)

    # override
    parser.add_argument('--seed', type=int, default=100, help='...')
    parser.add_argument('--dataset', type=str, default='freesolv', help='...')
    parser.add_argument('--epochs', type=int, default=1, help='...')
    parser.add_argument('--batch-size', type=int, default=None, help='...')
    parser.add_argument('--k', type=int, default=5, help='...')

    parser.add_argument('--backbone', type=str, default='GCN', help='...')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'mae', 'smooth'], help='...')

    return parser


def parse_args():
    parser = get_parser()
    p, _ = parser.parse_known_args()

    if getattr(p, 'config', None):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        
        valid_keys = set(vars(p).keys())
        
        for k in default_arg.keys():
            if k not in valid_keys:
                raise ValueError(f'WRONG ARG in YAML: {k} (not defined in parser)')

        parser.set_defaults(**default_arg)
    
    return parser.parse_args()
    

def main():
    args = parse_args()
    SET_SEED(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # DATA LOAD
    spec = get_dataset_spec(args.dataset)
    dataset = spec.reader(args.dataset_path + args.dataset + '.csv')
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = args.seed)

    if not bs("--batch-size"):
        if spec.default_batch_size is not None:
            args.batch_size = spec.default_batch_size

    num_desc_2d = spec.num_desc_2d
    collate_fn, num_desc_3d = build_collate_fn(args.dataset)

    # DEFINE THE MODEL
    from model import KROVEX, KROVEX_GCNs
    KROVEX = KROVEX.Net(dim_atomic_feat, num_desc_2d).to(device)
    KROVEX_GCNs = KROVEX_GCNs.Net(dim_atomic_feat, num_desc_2d).to(device)

    from model import test0219, test0220
    LapPE0219 = test0219.Net(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)
    LapPE0220 = test0220.Net(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)
    
    from model import test0221
    LapPE0221_2d = test0221.Net_2d(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)
    LapPE0221_3d = test0221.Net_3d(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)
    LapPE0221_total = test0221.Net_total(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)

    from model import Bilinear_Form
    BF0223_2d = Bilinear_Form.Net_2d(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)
    BF0223_3d = Bilinear_Form.Net_3d(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)
    BF0223_total = Bilinear_Form.Net(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)

    from model import GatedAttn_0224, GatedAttn_0224_2
    GatedAttn_2d = GatedAttn_0224.Net_2d(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)

    GatedAttn_2d_2 = GatedAttn_0224_2.Net_2d(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)
    GatedAttn_3d_2 = GatedAttn_0224_2.Net_3d(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)
    GatedAttn_total_2 = GatedAttn_0224_2.Net_total(dim_atomic_feat, num_desc_2d, num_desc_3d).to(device)


    # LOSS FUNC
    criterion = select_loss(args.loss)

    test_losses = dict()
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')

    # -------------------------- Baseline ------------------------------ #
    # # KROVEX
    # test_losses['KROVEX'], test_losses['KROVEX_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, KROVEX, criterion, args.batch_size, args.epochs, collate_fn, model_name='KROVEX_GCNs')
    # print(f'Final test | loss: ' + str(test_losses['KROVEX']) + '| R2: ' + str(test_losses['KROVEX_R2']))

    # # KROVEX GCN 직접 구현
    # test_losses['KROVEX_GCNs'], test_losses['KROVEX_GCNs_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, KROVEX_GCNs, criterion, args.batch_size, args.epochs, collate_fn, model_name='KROVEX_GCNs')
    # print(f'Final test | loss: ' + str(test_losses['KROVEX_GCNs']) + '| R2: ' + str(test_losses['KROVEX_GCNs_R2']))
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # # LapPE
    # test_losses['LapPE0219'], test_losses['LapPE0219_R2'] = evaluation_LapPE.full_train_and_test(train_dataset, test_dataset, LapPE0219, criterion, args.batch_size, args.epochs, collate_fn, model_name='LapPE0219')
    # print(f'Final test | loss: ' + str(test_losses['LapPE0219']) + '| R2: ' + str(test_losses['LapPE0219_R2']))

    # test_losses['LapPE0220'], test_losses['LapPE0220_R2'] = evaluation_LapPE.full_train_and_test(train_dataset, test_dataset, LapPE0220, criterion, args.batch_size, args.epochs, collate_fn, model_name='LapPE0220')
    # print(f'Final test | loss: ' + str(test_losses['LapPE0220']) + '| R2: ' + str(test_losses['LapPE0220_R2']))
    # ------------------------------------------------------------------ #
    

    # ---------------- Low-rank Tensor Fusion ----------------------------- #
    # test_losses['LapPE0221_2d'], test_losses['LapPE0221_2d_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, LapPE0221_2d, criterion, args.batch_size, args.epochs, collate_fn, model_name='LapPE0221_2d')
    # print(f'Final test | loss: ' + str(test_losses['LapPE0221_2d']) + '| R2: ' + str(test_losses['LapPE0221_2d_R2']))

    # test_losses['LapPE0221_3d'], test_losses['LapPE0221_3d_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, LapPE0221_3d, criterion, args.batch_size, args.epochs, collate_fn, model_name='LapPE0221_3d')
    # print(f'Final test | loss: ' + str(test_losses['LapPE0221_3d']) + '| R2: ' + str(test_losses['LapPE0221_3d_R2']))

    # test_losses['LapPE0221_total'], test_losses['LapPE0221_total_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, LapPE0221_total, criterion, args.batch_size, args.epochs, collate_fn, model_name='LapPE0221_total')
    # print(f'Final test | loss: ' + str(test_losses['LapPE0221_total']) + '| R2: ' + str(test_losses['LapPE0221_total_R2']))
    # ------------------------------------------------------------------ #

    # ---------------- Bilinear Form ----------------------------- #
    # 성능 good
    # test_losses['BF0223_2d'], test_losses['BF0223_2d_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, BF0223_2d, criterion, args.batch_size, args.epochs, collate_fn, model_name='BF0223_2d')
    # print(f'Final test | loss: ' + str(test_losses['BF0223_2d']) + '| R2: ' + str(test_losses['BF0223_2d_R2']))

    # test_losses['BF0223_3d'], test_losses['BF0223_3d_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, BF0223_3d, criterion, args.batch_size, args.epochs, collate_fn, model_name='BF0223_3d')
    # print(f'Final test | loss: ' + str(test_losses['BF0223_3d']) + '| R2: ' + str(test_losses['BF0223_3d_R2']))

    # test_losses['BF0223_total'], test_losses['BF0223_total_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, BF0223_total, criterion, args.batch_size, args.epochs, collate_fn, model_name='BF0223_total')
    # print(f'Final test | loss: ' + str(test_losses['BF0223_total']) + '| R2: ' + str(test_losses['BF0223_total_R2']))

    # ------------------------------------------------------------------ #

    # ---------------- Gated Attention ----------------------------- #
    # # Bilinear Form보다 성능 good 근데 의미가;
    # test_losses['GatedAttn_2d'], test_losses['GatedAttn_2d_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, GatedAttn_2d, criterion, args.batch_size, args.epochs, collate_fn, model_name='GatedAttn_2d')
    # print(f'Final test | loss: ' + str(test_losses['GatedAttn_2d']) + '| R2: ' + str(test_losses['GatedAttn_2d_R2']))


    # test_losses['GatedAttn_2d_2'], test_losses['GatedAttn_2d_2_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, GatedAttn_2d_2, criterion, args.batch_size, args.epochs, collate_fn, model_name='GatedAttn_2d_2')
    # print(f'Final test | loss: ' + str(test_losses['GatedAttn_2d_2']) + '| R2: ' + str(test_losses['GatedAttn_2d_2_R2']))

    # test_losses['GatedAttn_3d_2'], test_losses['GatedAttn_3d_2_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, GatedAttn_3d_2, criterion, args.batch_size, args.epochs, collate_fn, model_name='GatedAttn_3d_2')
    # print(f'Final test | loss: ' + str(test_losses['GatedAttn_3d_2']) + '| R2: ' + str(test_losses['GatedAttn_3d_2_R2']))

    test_losses['GatedAttn_total_2'], test_losses['GatedAttn_total_2_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, GatedAttn_total_2, criterion, args.batch_size, args.epochs, collate_fn, model_name='GatedAttn_total_2')
    print(f'Final test | loss: ' + str(test_losses['GatedAttn_total_2']) + '| R2: ' + str(test_losses['GatedAttn_total_2_R2']))

    # ------------------------------------------------------------------ #


    print('test_losses:', test_losses)
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')


if __name__ == '__main__':
    main()