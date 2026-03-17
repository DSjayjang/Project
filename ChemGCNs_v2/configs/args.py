import argparse
import yaml

def get_parser():
    parser = argparse.ArgumentParser(description='MODEL: KROVEX')
    
    parser.add_argument('--config', default='./configs/config.yaml')
    parser.add_argument('--r-home', default='C:/Programming/R/R-4.4.2', type=str) # Specify your path of 'R'
    parser.add_argument('--dataset-path', default='./datasets/', type=str)
    parser.add_argument('--phase', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--save-model', action='store_true')

    # override
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='freesolv')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=None,)
    parser.add_argument('--k', type=int, default=2)

    parser.add_argument('--backbone', type=str, default='GCN')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'mae'])
    parser.add_argument('--split', type=str, default='random', choices=['random', 'random_robust','scaffold', 'scaffold_robust', 'scaffold_robust2'])

    return parser