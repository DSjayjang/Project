import os
import yaml
import random
import numpy as np
import torch
import dgl


# ─── 1) config.yaml 파일 경로 계산 ───
# utils/ 아래에 있으므로, 두 단계 위(project root)로 올라가서 configs/config.yaml 참조
config_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'configs',
    'config.yaml'
)

# ─── 2) 실제 로드 ───
with open(config_path, 'r') as f:
    _cfg = yaml.safe_load(f)

SEED = _cfg['SEED']
DATASET = _cfg['DATASET']
BATCH_SIZE = _cfg['BATCH_SIZE']
MAX_EPOCHS = _cfg['MAX_EPOCHS']
K = _cfg['K']

def SET_SEED(seed: int = SEED):
    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(SEED)

    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
