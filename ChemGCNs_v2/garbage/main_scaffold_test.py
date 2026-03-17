import yaml
import argparse

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from utils.utils import SET_SEED, select_loss
from configs.registry import get_dataset_spec, build_collate_fn, bs
from utils.mol_props import dim_atomic_feat
from utils import evaluation_scaffold
from configs.args import get_parser

# import deepchem as dc
# from deepchem.splits import ScaffoldSplitter
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np

def get_scaffold_groups(smiles_list):
    scaffold_dict = {}

    for idx, s in enumerate(smiles_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s)

        if scaffold not in scaffold_dict:
            scaffold_dict[scaffold] = []

        scaffold_dict[scaffold].append(idx)

    print('scaffold_groups:', scaffold_dict)

    # groups = list(scaffold_dict.values())

    # return groups
    return scaffold_dict


# def scaffold_split(smiles_list, train_size=0.8):
#     scaffold_dict = get_scaffold_groups(smiles_list)

#     groups = list(scaffold_dict.values())
#     groups = sorted(groups, key=len, reverse=True)

#     n_total = len(smiles_list)
#     n_train_target = int(train_size * n_total)
    
#     train_idx, test_idx = [], []
#     train_scaffolds = []

#     for g in groups:
#         if len(train_idx)+len(g) <= n_train_target:
#             train_idx.extend(g)
#         else:
#             test_idx.extend(g)

#     return np.array(train_idx), np.array(test_idx)

def scaffold_split(smiles_list, train_size=0.8):

    scaffold_dict = get_scaffold_groups(smiles_list)

    # (scaffold, indices) 쌍으로 정렬
    scaffold_items = sorted(
        scaffold_dict.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    n_total = len(smiles_list)
    n_train_target = int(train_size * n_total)

    train_idx, test_idx = [], []
    train_scaffolds = []

    for scaffold, group in scaffold_items:

        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
            train_scaffolds.append(scaffold)
        else:
            test_idx.extend(group)

    # 🔥 train scaffold → id mapping
    scaffold2id = {scaf: i for i, scaf in enumerate(train_scaffolds)}

    return np.array(train_idx), np.array(test_idx), scaffold2id


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
        n_scaffolds_train = None
        scaffold2id = None
    
    elif args.split == 'scaffold':
        # train_idx, test_idx = scaffold_split(smiles_list)
        # train_dataset = [dataset[i] for i in train_idx]
        # test_dataset  = [dataset[i] for i in test_idx]

        # print('len(train_dataset)', len(train_dataset))
        # print('len(test_dataset)', len(test_dataset))
        train_idx, test_idx, scaffold2id = scaffold_split(smiles_list)

        train_dataset = [dataset[i] for i in train_idx]
        test_dataset  = [dataset[i] for i in test_idx]

        n_scaffolds_train = len(scaffold2id)

        # # print('len(train_dataset)', len(train_dataset))
        # # print('len(test_dataset)', len(test_dataset))
        # # print('n_scaffolds_train', n_scaffolds_train)
        # 🔥 전체 scaffold 리스트 생성
        all_scaffolds = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s) 
                        for s in smiles_list]

        # 🔥 train용 scaffold_id 생성
        train_scaffold_ids = []
        for i in train_idx:
            scaf = all_scaffolds[i]
            train_scaffold_ids.append(scaffold2id[scaf])

        print("n_scaffolds_train:", len(scaffold2id))
        train_dataset_with_scaf = []

        for data_item, scaf_id in zip(train_dataset, train_scaffold_ids):
            mol_graph, label = data_item              # ✅ 2개만 꺼낸다
            
            train_dataset_with_scaf.append((mol_graph, label, scaf_id))  # ✅ 3개로 확장

        train_dataset = train_dataset_with_scaf



    if not bs("--batch-size"):
        if spec.default_batch_size is not None:
            args.batch_size = spec.default_batch_size

    num_desc = spec.num_desc
    collate_fn = build_collate_fn(args.dataset)

    # DEFINE THE MODEL
    from model import KROVEX, KROVEX_GCNs
    # KROVEX = KROVEX.Net(dim_atomic_feat, num_desc).to(device)
    KROVEX_GCNs = KROVEX_GCNs.Net(dim_atomic_feat, num_desc).to(device)

    from model import BAN_scaffold_base, BAN_scaffold_cont
    ban = BAN_scaffold_base.Net(dim_atomic_feat, num_desc).to(device)
    ban_cont = BAN_scaffold_cont.Net(dim_atomic_feat, num_desc, n_scaffolds_train).to(device)

    # LOSS FUNC
    criterion = select_loss(args.loss)

    test_losses = dict()
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')

    # -------------------------- Baseline ------------------------------ #
    # # KROVEX
    # test_losses['KROVEX'], test_losses['KROVEX_R2'] = evaluation.evaluation(train_dataset, test_dataset, KROVEX, criterion, args.batch_size, args.epochs, collate_fn, model_name='KROVEX_GCNs')
    # print(f'Final test | loss: ' + str(test_losses['KROVEX']) + '| R2: ' + str(test_losses['KROVEX_R2']))

    # # KROVEX GCN 직접 구현
    # test_losses['KROVEX_GCNs'], test_losses['KROVEX_GCNs_R2'] = evaluation_scaffold.evaluation(train_dataset, test_dataset, KROVEX_GCNs, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='KROVEX_GCNs')
    # print(f'Final test | loss: ' + str(test_losses['KROVEX_GCNs']) + '| R2: ' + str(test_losses['KROVEX_GCNs_R2']))
    # ------------------------------------------------------------------ #


    # ----------------------------Bilinear Attention ----------------------------- #
    # Bilinear Attention
    test_losses['ban'], test_losses['ban_R2'] = evaluation_scaffold.evaluation(train_dataset, test_dataset, ban, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='ban')
    print(f'Final test | loss: ' + str(test_losses['ban']) + '| R2: ' + str(test_losses['ban_R2']))

    # # Bilinear Attention
    # test_losses['ban'], test_losses['ban_R2'] = evaluation_scaffold.evaluation(train_dataset, test_dataset, ban, criterion, desc_list, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path, model_name='ban')
    # print(f'Final test | loss: ' + str(test_losses['ban']) + '| R2: ' + str(test_losses['ban_R2']))

    # ------------------------------------------------------------------ #
    

    print('test_losses:', test_losses)
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')


if __name__ == '__main__':
    main()