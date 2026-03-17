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

from utils.scaffold import scaffold_split, scaffold_info
import numpy as np


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
    
    dataset, desc_list, smiles_list = spec.reader(args.dataset_path + args.dataset + '.csv')

    if args.split == 'random':
        train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = args.seed)
        print('len(train_dataset)', len(train_dataset))
        print('len(test_dataset)', len(test_dataset))
        num_scaffolds=None

    elif args.split == 'scaffold':
        train_idx, test_idx = scaffold_split(smiles_list)
 
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset  = [dataset[i] for i in test_idx]
        print('len(train_dataset)', len(train_dataset))
        print('len(test_dataset)', len(test_dataset))
        _, scaffold_ids, num_scaffolds, scaffold_to_id = scaffold_info(smiles_list)
        print("scaffold_ids:", scaffold_ids)
        print("num_scaffolds:", num_scaffolds)
        print("scaffold_to_id:", scaffold_to_id)

    if not bs("--batch-size"):
        if spec.default_batch_size is not None:
            args.batch_size = spec.default_batch_size

    num_desc = spec.num_desc
    collate_fn = build_collate_fn(args.dataset)

    # DEFINE THE MODEL
    from model import KROVEX, KROVEX_GCNs, GCN, GAT, GIN, GraphSAGE
    # KROVEX = KROVEX.Net(dim_atomic_feat, num_desc).to(device)
    KROVEX_GCNs = KROVEX_GCNs.Net(dim_atomic_feat, num_desc).to(device)

    from model import FABIG
    fag = FABIG.Net(dim_atomic_feat, num_desc).to(device)

    # LOSS FUNC
    criterion = select_loss(args.loss)

    test_losses = dict()
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')

    # # -------------------------- Baseline ------------------------------ #
    # # KROVEX GCN 직접 구현
    # test_losses['KROVEX_GCNs'], test_losses['KROVEX_GCNs_R2'] = evaluation.evaluation(train_dataset, test_dataset, KROVEX_GCNs, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='KROVEX_GCNs')
    # print(f'Final test | loss: ' + str(test_losses['KROVEX_GCNs']) + '| R2: ' + str(test_losses['KROVEX_GCNs_R2']))
 
    # total_params = sum(p.numel() for p in KROVEX_GCNs.parameters() if p.requires_grad)
    # param_mem = sum(p.numel() for p in KROVEX_GCNs.parameters()) * 4 / 1024**2
    # print(f"KROVEX 총 학습 가능한 파라미터 수: {total_params:,}")
    # print(f"Model parameter memory: {param_mem:.2f} MB")

    # ----------------------------FaBiG ----------------------------- #
    test_losses['fag'], test_losses['fag_R2'] = evaluation.evaluation(train_dataset, test_dataset, fag, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='fag')
    print(f'Final test | loss: ' + str(test_losses['fag']) + '| R2: ' + str(test_losses['fag_R2']))

    total_params = sum(p.numel() for p in fag.parameters() if p.requires_grad)
    param_mem = sum(p.numel() for p in fag.parameters()) * 4 / 1024**2
    print(f"fag 총 학습 가능한 파라미터 수: {total_params:,}")
    print(f"Model parameter memory: {param_mem:.2f} MB")

    print('test_losses:', test_losses)
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')


if __name__ == '__main__':
    main()