import yaml
import argparse

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from utils.utils import SET_SEED, select_loss
from configs.registry import get_dataset_spec, build_collate_fn, bs
from utils.mol_props import dim_atomic_feat
from utils import evaluation

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
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'mae'], help='...')

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

    if not bs("--batch-size"):
        if spec.default_batch_size is not None:
            args.batch_size = spec.default_batch_size

    num_desc_2d = spec.num_desc_2d
    collate_fn, num_desc_3d = build_collate_fn(args.dataset)

    # DEFINE THE MODEL
    from model import KROVEX_GCNs
    KROVEX_GCNs = KROVEX_GCNs.Net(dim_atomic_feat, num_desc_2d).to(device)

    # LOSS FUNC
    criterion = select_loss(args.loss)

    test_losses = dict()
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')

    # -------------------------- Baseline ------------------------------ #
    # # KROVEX
    # val_losses['KROVEX'] = cv.cross_validation(train_dataset, KROVEX, criterion, K, BATCH_SIZE, MAX_EPOCHS, collate_fn, model_name='KROVEX')
    # print('CV loss (KROVEX): ' + str(val_losses['KROVEX']))

    # KROVEX GCN 직접 구현
    test_losses['KROVEX_GCNs'], test_losses['KROVEX_GCNs_R2'] = evaluation.full_train_and_test(train_dataset, test_dataset, KROVEX_GCNs, criterion, args.batch_size, args.epochs, collate_fn, model_name='KROVEX_GCNs')
    print(f'Final test | loss: ' + str(test_losses['KROVEX_GCNs']) + '| R2: ' + str(test_losses['KROVEX_GCNs_R2']))
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    
    print('test_losses:', test_losses)
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')


if __name__ == '__main__':
    main()