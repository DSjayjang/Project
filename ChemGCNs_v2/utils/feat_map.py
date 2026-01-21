import numpy as np
import pandas as pd

from configs.config import DATASET_PATH
from utils.utils import Z_Score_new

def selected_descriptors_scgas():
    desc_list = ['smiles', 'use_MORSE_129', 'use_MORSE_161', 'use_GETAWAY_105', 'use_GETAWAY_133', 'use_GETAWAY_025', 'use_RadiusOfGyration', 'use_MORSE_007', 'use_GETAWAY_085', 'use_GETAWAY_263', 'use_GETAWAY_086', 'use_MORSE_194', 'use_GETAWAY_045', 'use_MORSE_065', 'use_MORSE_193', 'use_MORSE_094', 'use_USRCAT_057', 'use_USRCAT_000', 'use_GETAWAY_005', 'use_GETAWAY_093', 'use_GETAWAY_007', 'use_USRCAT_007', 'use_GETAWAY_002', 'use_GETAWAY_257', 'use_USR_006', 'use_USRCAT_054', 'use_GETAWAY_113', 'use_MORSE_068', 'use_MORSE_097', 'use_MORSE_033', 'use_GETAWAY_067', 'use_GETAWAY_107', 'use_GETAWAY_047', 'use_USRCAT_058', 'use_MORSE_222', 'use_USRCAT_055', 'use_MORSE_167', 'use_USRCAT_043', 'use_GETAWAY_046', 'use_GETAWAY_000', 'use_USRCAT_052', 'use_MORSE_135', 'use_GETAWAY_066', 'use_USRCAT_051', 'use_GETAWAY_084']
    
    return desc_list

def build_feat_map(dataset):
    if dataset == 'freesolv':
        print('데이터셋 확인')
    elif dataset == 'scgas':
        df3d = pd.read_csv(r'.\datasets\features_all1_drop_zeros.csv')
        df3d = df3d[selected_descriptors_scgas()]

    smiles_col = 'smiles'
    fillna_value = 0.0

    feature_cols = [c for c in df3d.columns if c != smiles_col]

    df = df3d[[smiles_col] + feature_cols].copy()
    # df = df.drop_duplicates(smiles_col)

    # Normalization
    X = df[feature_cols].values.astype(np.float32)
    X = Z_Score_new(X)
    df[feature_cols] = X

    # 결측치는 어떻게 처리할지 고민 필요
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(fillna_value)

    # smiles -> np.array(D3,)
    feat_map = dict(zip(
        df[smiles_col].values,
        df[feature_cols].values.astype(np.float32)))

    return feat_map, feature_cols