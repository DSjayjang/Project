import numpy as np
import pandas as pd

from configs.config import DATASET_PATH

def selected_descriptors_scgas():
    desc_list = ['smiles', 'use_GETAWAY_003', 'use_GETAWAY_063', 'use_MORSE_129', 'use_USRCAT_057',
       'use_GETAWAY_002', 'use_GETAWAY_165', 'use_MORSE_145',
       'use_AUTOCORR3D_020', 'use_MORSE_163', 'use_USRCAT_054',
       'use_MORSE_223', 'use_MORSE_194', 'use_AUTOCORR3D_010', 'use_MORSE_130',
       'use_USRCAT_033', 'use_GETAWAY_047', 'use_GETAWAY_142',
       'use_Eccentricity', 'use_USRCAT_058', 'use_USRCAT_055',
       'use_GETAWAY_093', 'use_MORSE_058', 'use_MORSE_203', 'use_PMI3',
       'use_GETAWAY_055', 'use_USRCAT_025', 'use_MORSE_105', 'use_NPR2',
       'use_MORSE_207', 'use_GETAWAY_137', 'use_GETAWAY_028',
       'use_AUTOCORR3D_013', 'use_GETAWAY_086', 'use_MORSE_190',
       'use_GETAWAY_177', 'use_MORSE_143', 'use_GETAWAY_145', 'use_MORSE_183',
       'use_MORSE_148']
    
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

    # 결측치는 어떻게 처리할지 고민 필요
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(fillna_value)

    # smiles -> np.array(D3,)
    feat_map = dict(zip(
        df[smiles_col].values,
        df[feature_cols].values.astype(np.float32)))

    return feat_map, feature_cols