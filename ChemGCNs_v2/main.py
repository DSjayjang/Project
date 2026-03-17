import yaml
import argparse

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from utils.utils import SET_SEED, select_loss
from configs.registry import get_dataset_spec, build_collate_fn, bs
from utils.mol_props import dim_atomic_feat
from utils import evaluation, evaluation_scaffold_robust, evaluation_scaffold_robust2, evaluation_scaffold_robust3, evaluation_scaffold_robust4_0307
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
    gcn = GCN.Net(dim_atomic_feat).to(device)
    gat = GAT.Net(dim_atomic_feat).to(device)
    gin = GIN.Net(dim_atomic_feat).to(device)
    gsg = GraphSAGE.Net(dim_atomic_feat).to(device)

    from model import BAN, BAN_robust,BAN_robust2, BAN_robust3, BAN_robust4_0307
    ban = BAN.Net(dim_atomic_feat, num_desc).to(device)
    # ban_robust = BAN_robust.Net(dim_atomic_feat, num_desc, num_scaffolds).to(device)
    # ban_robust2 = BAN_robust2.Net(dim_atomic_feat, num_desc, num_scaffolds).to(device)
    ban_robust3 = BAN_robust3.Net(dim_atomic_feat, num_desc, num_scaffolds).to(device)
    ban_robust4 = BAN_robust4_0307.Net(dim_atomic_feat, num_desc, num_scaffolds).to(device)


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

    # # -------------------------- GCN ------------------------------ #
    # # Graph Convolutional Networks
    # test_losses['GCN'], test_losses['GCN_R2'] = evaluation.evaluation(train_dataset, test_dataset, gcn, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='GCN')
    # print(f'Final test | loss: ' + str(test_losses['GCN']) + '| R2: ' + str(test_losses['GCN_R2']))
 
    # total_params = sum(p.numel() for p in gcn.parameters() if p.requires_grad)
    # param_mem = sum(p.numel() for p in gcn.parameters()) * 4 / 1024**2
    # print(f"GCN 총 학습 가능한 파라미터 수: {total_params:,}")
    # print(f"Model parameter memory: {param_mem:.2f} MB")

    # # -------------------------- GAT ------------------------------ #
    # # Graph Attention Networks
    # test_losses['GAT'], test_losses['GAT_R2'] = evaluation.evaluation(train_dataset, test_dataset, gat, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='GAT')
    # print(f'Final test | loss: ' + str(test_losses['GAT']) + '| R2: ' + str(test_losses['GAT_R2']))
 
    # total_params = sum(p.numel() for p in gat.parameters() if p.requires_grad)
    # param_mem = sum(p.numel() for p in gat.parameters()) * 4 / 1024**2
    # print(f"GAT 총 학습 가능한 파라미터 수: {total_params:,}")
    # print(f"Model parameter memory: {param_mem:.2f} MB")

    # # -------------------------- GIN ------------------------------ #
    # # Graph Isomorphism Networks
    # test_losses['GIN'], test_losses['GIN_R2'] = evaluation.evaluation(train_dataset, test_dataset, gin, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='GIN')
    # print(f'Final test | loss: ' + str(test_losses['GIN']) + '| R2: ' + str(test_losses['GIN_R2']))
 
    # total_params = sum(p.numel() for p in gin.parameters() if p.requires_grad)
    # param_mem = sum(p.numel() for p in gin.parameters()) * 4 / 1024**2
    # print(f"GIN 총 학습 가능한 파라미터 수: {total_params:,}")
    # print(f"Model parameter memory: {param_mem:.2f} MB")

    # # -------------------------- GraphSAGE ------------------------------ #
    # # GraphSAGE
    # test_losses['GraphSAGE'], test_losses['GraphSAGE_R2'] = evaluation.evaluation(train_dataset, test_dataset, gsg, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='GraphSAGE')
    # print(f'Final test | loss: ' + str(test_losses['GraphSAGE']) + '| R2: ' + str(test_losses['GraphSAGE_R2']))
 
    # total_params = sum(p.numel() for p in gsg.parameters() if p.requires_grad)
    # param_mem = sum(p.numel() for p in gsg.parameters()) * 4 / 1024**2
    # print(f"GraphSAGE 총 학습 가능한 파라미터 수: {total_params:,}")
    # print(f"Model parameter memory: {param_mem:.2f} MB")

    # ----------------------------BAN ----------------------------- #
    # Bilinear Attention Networks
    test_losses['ban'], test_losses['ban_R2'] = evaluation.evaluation(train_dataset, test_dataset, ban, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='ban')
    print(f'Final test | loss: ' + str(test_losses['ban']) + '| R2: ' + str(test_losses['ban_R2']))

    total_params = sum(p.numel() for p in ban.parameters() if p.requires_grad)
    param_mem = sum(p.numel() for p in ban.parameters()) * 4 / 1024**2
    print(f"BAN 총 학습 가능한 파라미터 수: {total_params:,}")
    print(f"Model parameter memory: {param_mem:.2f} MB")









    #
    # SCAFFOLD ROBUST
    #



    # # ----------------------------BAN + scaffold robust ----------------------------- #
    # # Bilinear Attention Networks + scaffold robust
    # test_losses['BAN_robust'], test_losses['BAN_robust_R2'] = evaluation_scaffold_robust.evaluation(train_dataset, test_dataset, ban_robust, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='ban_robust')
    # print(f'Final test | loss: ' + str(test_losses['BAN_robust']) + '| R2: ' + str(test_losses['BAN_robust_R2']))

    # total_params = sum(p.numel() for p in ban_robust.parameters() if p.requires_grad)
    # param_mem = sum(p.numel() for p in ban_robust.parameters()) * 4 / 1024**2
    # print(f"BAN_robust 총 학습 가능한 파라미터 수: {total_params:,}")
    # print(f"Model parameter memory: {param_mem:.2f} MB")

    # # ----------------------------BAN + scaffold robust2 ----------------------------- #
    # # Bilinear Attention Networks + scaffold robust
    # test_losses['BAN_robust2'], test_losses['BAN_robust2_R2'] = evaluation_scaffold_robust2.evaluation(train_dataset, test_dataset, ban_robust2, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='ban_robust2')
    # print(f'Final test | loss: ' + str(test_losses['BAN_robust2']) + '| R2: ' + str(test_losses['BAN_robust2_R2']))

    # total_params = sum(p.numel() for p in ban_robust2.parameters() if p.requires_grad)
    # param_mem = sum(p.numel() for p in ban_robust2.parameters()) * 4 / 1024**2
    # print(f"BAN_robust 총 학습 가능한 파라미터 수: {total_params:,}")
    # print(f"Model parameter memory: {param_mem:.2f} MB")

    # # ----------------------------BAN + scaffold robust3 ----------------------------- #
    # # Bilinear Attention Networks + scaffold robust
    # test_losses['BAN_robust3'], test_losses['BAN_robust3_R2'] = evaluation_scaffold_robust3.evaluation(train_dataset, test_dataset, ban_robust3, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='ban_robust3')
    # print(f'Final test | loss: ' + str(test_losses['BAN_robust3']) + '| R2: ' + str(test_losses['BAN_robust3_R2']))

    # total_params = sum(p.numel() for p in ban_robust3.parameters() if p.requires_grad)
    # param_mem = sum(p.numel() for p in ban_robust3.parameters()) * 4 / 1024**2
    # print(f"BAN_robust3 총 학습 가능한 파라미터 수: {total_params:,}")
    # print(f"Model parameter memory: {param_mem:.2f} MB")

    # # ----------------------------BAN + scaffold robust4 ----------------------------- #
    # # Bilinear Attention Networks + scaffold robust
    # test_losses['BAN_robust4'], test_losses['BAN_robust4_R2'] = evaluation_scaffold_robust4_0307.evaluation(train_dataset, test_dataset, ban_robust4, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='ban_robust4')
    # print(f'Final test | loss: ' + str(test_losses['BAN_robust4']) + '| R2: ' + str(test_losses['BAN_robust4_R2']))

    # total_params = sum(p.numel() for p in ban_robust4.parameters() if p.requires_grad)
    # param_mem = sum(p.numel() for p in ban_robust4.parameters()) * 4 / 1024**2
    # print(f"BAN_robust4 총 학습 가능한 파라미터 수: {total_params:,}")
    # print(f"Model parameter memory: {param_mem:.2f} MB")



    print('test_losses:', test_losses)
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')


if __name__ == '__main__':
    main()