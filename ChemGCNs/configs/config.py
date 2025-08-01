import os
import yaml
import random
import numpy as np
import torch
import dgl

# YAML 파일 경로를 BASE_DIR 아래에서 찾도록 설정
with open('configs\config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# cfg_path = os.path.join(BASE_DIR, "config.yaml")

# with open(cfg_path, 'r') as f:
#     cfg = yaml.safe_load(f)


SEED = cfg['SEED']
DATASET = cfg['DATASET']
BATCH_SIZE = cfg['BATCH_SIZE']
MAX_EPOCHS = cfg['MAX_EPOCHS']
K = cfg['K']
R_HOME = cfg['R_HOME']

def SET_SEED():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dgl.random.seed(SEED)

    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['R_HOME'] = cfg['R_HOME'] # for ISIS

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
