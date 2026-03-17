import yaml
import argparse

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from utils.utils import SET_SEED, select_loss
from configs.registry import get_dataset_spec, build_collate_fn, bs
from utils.mol_props import dim_atomic_feat
from utils import evaluation
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
    
    dataset, desc_list, _ = spec.reader(args.dataset_path + args.dataset + '.csv')
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = args.seed)

    if not bs("--batch-size"):
        if spec.default_batch_size is not None:
            args.batch_size = spec.default_batch_size

    num_desc = spec.num_desc
    collate_fn = build_collate_fn(args.dataset)

    # DEFINE THE MODEL
    from model import KROVEX, KROVEX_GCNs
    # KROVEX = KROVEX.Net(dim_atomic_feat, num_desc).to(device)
    KROVEX_GCNs = KROVEX_GCNs.Net(dim_atomic_feat, num_desc).to(device)

    from model import BAN
    ban = BAN.Net(dim_atomic_feat, num_desc).to(device)

    # LOSS FUNC
    criterion = select_loss(args.loss)

    test_losses = dict()
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')

    # -------------------------- Baseline ------------------------------ #
    # # KROVEX
    # test_losses['KROVEX'], test_losses['KROVEX_R2'] = evaluation.evaluation(train_dataset, test_dataset, KROVEX, criterion, args.batch_size, args.epochs, collate_fn, model_name='KROVEX_GCNs')
    # print(f'Final test | loss: ' + str(test_losses['KROVEX']) + '| R2: ' + str(test_losses['KROVEX_R2']))

    # # KROVEX GCN 직접 구현
    # test_losses['KROVEX_GCNs'], test_losses['KROVEX_GCNs_R2'] = evaluation.evaluation(train_dataset, test_dataset, KROVEX_GCNs, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='KROVEX_GCNs')
    # print(f'Final test | loss: ' + str(test_losses['KROVEX_GCNs']) + '| R2: ' + str(test_losses['KROVEX_GCNs_R2']))
    # ------------------------------------------------------------------ #


    # ----------------------------Bilinear Attention ----------------------------- #
    # Bilinear Attention
    test_losses['ban'], test_losses['ban_R2'] = evaluation.evaluation(train_dataset, test_dataset, ban, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='ban')
    print(f'Final test | loss: ' + str(test_losses['ban']) + '| R2: ' + str(test_losses['ban_R2']))

    # ------------------------------------------------------------------ #
    

    print('test_losses:', test_losses)
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')


if __name__ == '__main__':
    main()