import numpy as np
import pandas as pd

from configs.config import DATASET_PATH
from utils.utils import Z_Score_new

def selected_descriptors_scgas():
    desc_list = ['smiles', 'use_GETAWAY_044', 'use_GETAWAY_002', 'use_GETAWAY_105', 'use_USRCAT_057', 'use_GETAWAY_076', 'use_GETAWAY_093', 'use_MORSE_129', 'use_MORSE_145', 'use_MORSE_211', 'use_GETAWAY_187', 'use_GETAWAY_136', 'use_AUTOCORR3D_074', 'use_GETAWAY_074', 'use_USRCAT_054', 'use_MORSE_033', 'use_GETAWAY_059', 'use_GETAWAY_086', 'use_GETAWAY_025', 'use_USRCAT_025', 'use_AUTOCORR3D_062', 'use_MORSE_161', 'use_MORSE_194', 'use_USRCAT_052', 'use_MORSE_080', 'use_USRCAT_056', 'use_MORSE_203', 'use_GETAWAY_080', 'use_USRCAT_049', 'use_GETAWAY_236', 'use_MORSE_009', 'use_USRCAT_058', 'use_MORSE_117', 'use_GETAWAY_184', 'use_MORSE_193', 'use_GETAWAY_115', 'use_MORSE_108', 'use_GETAWAY_265', 'use_NPR2', 'use_MORSE_222', 'use_PMI3', 'use_AUTOCORR3D_065', 'use_GETAWAY_165', 'use_MORSE_149', 'use_AUTOCORR3D_021', 'use_MORSE_200', 'use_MORSE_132', 'use_GETAWAY_193', 'use_GETAWAY_028', 'use_MORSE_030', 'use_USRCAT_055', 'use_USRCAT_021', 'use_MORSE_130', 'use_GETAWAY_047', 'use_MORSE_003', 'use_AUTOCORR3D_000', 'use_GETAWAY_250', 'use_AUTOCORR3D_038', 'use_USRCAT_042', 'use_GETAWAY_033', 'use_MORSE_207', 'use_USRCAT_043', 'use_MORSE_089', 'use_USR_008', 'use_GETAWAY_110', 'use_MORSE_202', 'use_GETAWAY_000', 'use_GETAWAY_089', 'use_USRCAT_040', 'use_AUTOCORR3D_011', 'use_USRCAT_045', 'use_AUTOCORR3D_066', 'use_MORSE_189', 'use_MORSE_164', 'use_MORSE_143', 'use_USRCAT_022', 'use_MORSE_208', 'use_MORSE_148', 'use_USRCAT_035', 'use_MORSE_206', 'use_GETAWAY_167', 'use_USRCAT_023', 'use_GETAWAY_269', 'use_GETAWAY_166', 'use_GETAWAY_138', 'use_GETAWAY_177', 'use_AUTOCORR3D_072', 'use_GETAWAY_241', 'use_AUTOCORR3D_073', 'use_GETAWAY_070', 'use_GETAWAY_001', 'use_MORSE_094', 'use_AUTOCORR3D_018', 'use_GETAWAY_153', 'use_GETAWAY_030', 'use_MORSE_144', 'use_GETAWAY_189', 'use_AUTOCORR3D_012', 'use_GETAWAY_142', 'use_MORSE_150', 'use_MORSE_165', 'use_GETAWAY_256', 'use_MORSE_163', 'use_USRCAT_017', 'use_GETAWAY_051', 'use_GETAWAY_211', 'use_MORSE_048', 'use_USRCAT_047', 'use_GETAWAY_007', 'use_AUTOCORR3D_014', 'use_MORSE_197', 'use_AUTOCORR3D_061', 'use_USRCAT_044', 'use_GETAWAY_161', 'use_GETAWAY_270', 'use_USRCAT_050', 'use_Eccentricity', 'use_USRCAT_002', 'use_AUTOCORR3D_068', 'use_GETAWAY_261', 'use_GETAWAY_176', 'use_AUTOCORR3D_019', 'use_GETAWAY_208', 'use_GETAWAY_268', 'use_InertialShapeFactor', 'use_GETAWAY_257', 'use_GETAWAY_223', 'use_USRCAT_051', 'use_USRCAT_020', 'use_GETAWAY_029']
    
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