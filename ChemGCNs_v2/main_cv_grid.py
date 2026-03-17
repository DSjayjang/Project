import yaml
import argparse
from itertools import product

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from utils.utils import SET_SEED, select_loss
from configs.registry import get_dataset_spec, build_collate_fn, bs
from utils.mol_props import dim_atomic_feat
from utils import cv, cv_gridsearch
from configs.args import get_parser

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
    ckpt_path = spec.ckpt_path # saved model
    
    dataset, desc_list = spec.reader(args.dataset_path + args.dataset + '.csv')
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = args.seed)

    if not bs("--batch-size"):
        if spec.default_batch_size is not None:
            args.batch_size = spec.default_batch_size

    num_desc = spec.num_desc
    collate_fn = build_collate_fn(args.dataset)

    # LOSS FUNC
    criterion = select_loss(args.loss)

    val_losses = dict()

    # -------------------------- TEST GRID SEARCH ------------------------------ #
    grid = {
        "dim_graph": [20, 64, 128],
        "d_t": [32, 64, 128],
        "K": [32, 64, 128],
        # "k_attn_mult": [1, 2, 3],
        "glimpse": [1, 2, 4],        
        "fc1": [128, 256],
        "fc2": [32, 64],
    }

    param_list = [
        dict(zip(grid.keys(), vals))
        for vals in product(*grid.values())
    ]
    print("num configs:", len(param_list))
    # --------------------------  ------------------------------ #


    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')

    val_losses['ban'] = cv_gridsearch.grid_search_kfold(dataset, dim_atomic_feat, num_desc, param_list, criterion, args.k, args.batch_size, args.epochs, args.seed, args.dataset, collate_fn, model_name = 'ban')

    # print('val_losses:', val_losses)
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')


if __name__ == '__main__':
    main()