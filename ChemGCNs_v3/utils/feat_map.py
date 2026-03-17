import numpy as np
import pandas as pd

from utils.utils import Z_Score_new

def selected_descriptors_scgas():
    # seed 30
    # desc_list = ['smiles', 'use_MORSE_065', 'use_MORSE_129', 'use_GETAWAY_133', 'use_GETAWAY_045', 'use_MORSE_161', 'use_GETAWAY_263', 'use_GETAWAY_025', 'use_GETAWAY_105', 'use_GETAWAY_007', 'use_GETAWAY_107', 'use_GETAWAY_002', 'use_USRCAT_048', 'use_GETAWAY_086', 'use_MORSE_194', 'use_MORSE_222', 'use_GETAWAY_000', 'use_MORSE_193', 'use_USRCAT_055', 'use_GETAWAY_047', 'use_GETAWAY_155', 'use_GETAWAY_125', 'use_MORSE_164', 'use_USRCAT_058', 'use_USRCAT_034', 'use_USRCAT_031']

    # seed 50
    # desc_list = ['smiles', 'use_MORSE_129', 'use_MORSE_161', 'use_GETAWAY_105', 'use_GETAWAY_133', 'use_GETAWAY_025', 'use_RadiusOfGyration', 'use_MORSE_007', 'use_GETAWAY_085', 'use_GETAWAY_263', 'use_GETAWAY_086', 'use_MORSE_194', 'use_GETAWAY_045', 'use_MORSE_065', 'use_MORSE_193', 'use_MORSE_094', 'use_USRCAT_057', 'use_USRCAT_000', 'use_GETAWAY_005', 'use_GETAWAY_093', 'use_GETAWAY_007', 'use_USRCAT_007', 'use_GETAWAY_002', 'use_GETAWAY_257', 'use_USR_006', 'use_USRCAT_054', 'use_GETAWAY_113', 'use_MORSE_068', 'use_MORSE_097', 'use_MORSE_033', 'use_GETAWAY_067', 'use_GETAWAY_107', 'use_GETAWAY_047', 'use_USRCAT_058', 'use_MORSE_222', 'use_USRCAT_055', 'use_MORSE_167', 'use_USRCAT_043', 'use_GETAWAY_046', 'use_GETAWAY_000', 'use_USRCAT_052', 'use_MORSE_135', 'use_GETAWAY_066', 'use_USRCAT_051', 'use_GETAWAY_084']
    
    # seed 100
    desc_list = ['smiles', 'use_MORSE_129', 'use_AUTOCORR3D_070', 'use_MORSE_115', 'use_AUTOCORR3D_021', 'use_GETAWAY_184', 'use_AUTOCORR3D_020', 'use_GETAWAY_105', 'use_MORSE_147', 'use_GETAWAY_002', 'use_PMI3', 'use_MORSE_161', 'use_AUTOCORR3D_075', 'use_GETAWAY_104', 'use_AUTOCORR3D_030', 'use_AUTOCORR3D_071', 'use_GETAWAY_044', 'use_GETAWAY_074', 'use_GETAWAY_067', 'use_GETAWAY_238', 'use_MORSE_097', 'use_GETAWAY_060', 'use_GETAWAY_025', 'use_PMI2', 'use_GETAWAY_086', 'use_USRCAT_006', 'use_GETAWAY_136', 'use_USRCAT_057', 'use_MORSE_007', 'use_MORSE_193', 'use_GETAWAY_152', 'use_GETAWAY_045', 'use_RadiusOfGyration', 'use_AUTOCORR3D_051', 'use_GETAWAY_094', 'use_MORSE_141', 'use_MORSE_194', 'use_GETAWAY_187', 'use_USRCAT_054', 'use_MORSE_109', 'use_GETAWAY_220', 'use_MORSE_030', 'use_USRCAT_055', 'use_MORSE_100', 'use_USRCAT_043', 'use_AUTOCORR3D_026', 'use_GETAWAY_173', 'use_GETAWAY_144', 'use_GETAWAY_257', 'use_MORSE_068', 'use_GETAWAY_127', 'use_GETAWAY_076', 'use_USRCAT_058', 'use_USRCAT_025', 'use_MORSE_132', 'use_AUTOCORR3D_061', 'use_GETAWAY_202', 'use_GETAWAY_256', 'use_MORSE_131', 'use_MORSE_222', 'use_MORSE_184', 'use_MORSE_167', 'use_GETAWAY_080', 'use_PMI1', 'use_GETAWAY_014', 'use_GETAWAY_064', 'use_GETAWAY_073', 'use_AUTOCORR3D_041', 'use_MORSE_135', 'use_AUTOCORR3D_056', 'use_AUTOCORR3D_040', 'use_MORSE_164', 'use_USRCAT_051', 'use_MORSE_190', 'use_GETAWAY_116']
    
    # seed 130
    # desc_list = ['smiles', 'use_MORSE_129', 'use_GETAWAY_105', 'use_AUTOCORR3D_021', 'use_AUTOCORR3D_070', 'use_GETAWAY_044', 'use_MORSE_161', 'use_GETAWAY_184', 'use_AUTOCORR3D_074', 'use_AUTOCORR3D_020', 'use_MORSE_147', 'use_GETAWAY_002', 'use_AUTOCORR3D_071', 'use_GETAWAY_144', 'use_GETAWAY_238', 'use_MORSE_193', 'use_GETAWAY_187', 'use_GETAWAY_104', 'use_PMI3', 'use_MORSE_115', 'use_GETAWAY_045', 'use_AUTOCORR3D_075', 'use_GETAWAY_014', 'use_MORSE_100', 'use_MORSE_007', 'use_GETAWAY_060', 'use_MORSE_104', 'use_GETAWAY_255', 'use_GETAWAY_201', 'use_USRCAT_057', 'use_GETAWAY_136', 'use_MORSE_211', 'use_GETAWAY_074', 'use_PMI2', 'use_GETAWAY_024', 'use_GETAWAY_099', 'use_MORSE_097', 'use_GETAWAY_263', 'use_AUTOCORR3D_054', 'use_GETAWAY_076', 'use_MORSE_068', 'use_RadiusOfGyration', 'use_GETAWAY_007', 'use_MORSE_168', 'use_MORSE_132', 'use_GETAWAY_219', 'use_MORSE_072', 'use_USRCAT_003', 'use_GETAWAY_025', 'use_USR_003', 'use_AUTOCORR3D_024', 'use_AUTOCORR3D_051', 'use_USRCAT_054', 'use_AUTOCORR3D_000', 'use_USRCAT_055', 'use_USR_000', 'use_MORSE_194', 'use_GETAWAY_000', 'use_GETAWAY_193', 'use_GETAWAY_124', 'use_USR_001', 'use_USRCAT_025', 'use_AUTOCORR3D_061', 'use_USR_007', 'use_GETAWAY_093', 'use_USRCAT_007', 'use_GETAWAY_145', 'use_GETAWAY_006', 'use_AUTOCORR3D_030', 'use_GETAWAY_146', 'use_USR_006', 'use_GETAWAY_026', 'use_USRCAT_043', 'use_USRCAT_006', 'use_USRCAT_022', 'use_USRCAT_058', 'use_MORSE_094', 'use_AUTOCORR3D_046', 'use_GETAWAY_152', 'use_MORSE_199', 'use_GETAWAY_220', 'use_GETAWAY_083', 'use_MORSE_222', 'use_USRCAT_031', 'use_AUTOCORR3D_056', 'use_GETAWAY_095', 'use_MORSE_030', 'use_GETAWAY_015', 'use_GETAWAY_084', 'use_USR_010', 'use_PMI1', 'use_GETAWAY_073', 'use_GETAWAY_096', 'use_AUTOCORR3D_026', 'use_AUTOCORR3D_006', 'use_MORSE_164', 'use_GETAWAY_151', 'use_AUTOCORR3D_076', 'use_MORSE_173', 'use_MORSE_190', 'use_AUTOCORR3D_005']

    # seed all
    # desc_list = ['smiles', 'use_GETAWAY_044', 'use_GETAWAY_002', 'use_GETAWAY_105', 'use_USRCAT_057', 'use_GETAWAY_076', 'use_GETAWAY_093', 'use_MORSE_129', 'use_MORSE_145', 'use_MORSE_211', 'use_GETAWAY_187', 'use_GETAWAY_136', 'use_AUTOCORR3D_074', 'use_GETAWAY_074', 'use_USRCAT_054', 'use_MORSE_033', 'use_GETAWAY_059', 'use_GETAWAY_086', 'use_GETAWAY_025', 'use_USRCAT_025', 'use_AUTOCORR3D_062', 'use_MORSE_161', 'use_MORSE_194', 'use_USRCAT_052', 'use_MORSE_080', 'use_USRCAT_056', 'use_MORSE_203', 'use_GETAWAY_080', 'use_USRCAT_049', 'use_GETAWAY_236', 'use_MORSE_009', 'use_USRCAT_058', 'use_MORSE_117', 'use_GETAWAY_184', 'use_MORSE_193', 'use_GETAWAY_115', 'use_MORSE_108', 'use_GETAWAY_265', 'use_NPR2', 'use_MORSE_222', 'use_PMI3', 'use_AUTOCORR3D_065', 'use_GETAWAY_165', 'use_MORSE_149', 'use_AUTOCORR3D_021', 'use_MORSE_200', 'use_MORSE_132', 'use_GETAWAY_193', 'use_GETAWAY_028', 'use_MORSE_030', 'use_USRCAT_055', 'use_USRCAT_021', 'use_MORSE_130', 'use_GETAWAY_047', 'use_MORSE_003', 'use_AUTOCORR3D_000', 'use_GETAWAY_250', 'use_AUTOCORR3D_038', 'use_USRCAT_042', 'use_GETAWAY_033', 'use_MORSE_207', 'use_USRCAT_043', 'use_MORSE_089', 'use_USR_008', 'use_GETAWAY_110', 'use_MORSE_202', 'use_GETAWAY_000', 'use_GETAWAY_089', 'use_USRCAT_040', 'use_AUTOCORR3D_011', 'use_USRCAT_045', 'use_AUTOCORR3D_066', 'use_MORSE_189', 'use_MORSE_164', 'use_MORSE_143', 'use_USRCAT_022', 'use_MORSE_208', 'use_MORSE_148', 'use_USRCAT_035', 'use_MORSE_206', 'use_GETAWAY_167', 'use_USRCAT_023', 'use_GETAWAY_269', 'use_GETAWAY_166', 'use_GETAWAY_138', 'use_GETAWAY_177', 'use_AUTOCORR3D_072', 'use_GETAWAY_241', 'use_AUTOCORR3D_073', 'use_GETAWAY_070', 'use_GETAWAY_001', 'use_MORSE_094', 'use_AUTOCORR3D_018', 'use_GETAWAY_153', 'use_GETAWAY_030', 'use_MORSE_144', 'use_GETAWAY_189', 'use_AUTOCORR3D_012', 'use_GETAWAY_142', 'use_MORSE_150', 'use_MORSE_165', 'use_GETAWAY_256', 'use_MORSE_163', 'use_USRCAT_017', 'use_GETAWAY_051', 'use_GETAWAY_211', 'use_MORSE_048', 'use_USRCAT_047', 'use_GETAWAY_007', 'use_AUTOCORR3D_014', 'use_MORSE_197', 'use_AUTOCORR3D_061', 'use_USRCAT_044', 'use_GETAWAY_161', 'use_GETAWAY_270', 'use_USRCAT_050', 'use_Eccentricity', 'use_USRCAT_002', 'use_AUTOCORR3D_068', 'use_GETAWAY_261', 'use_GETAWAY_176', 'use_AUTOCORR3D_019', 'use_GETAWAY_208', 'use_GETAWAY_268', 'use_InertialShapeFactor', 'use_GETAWAY_257', 'use_GETAWAY_223', 'use_USRCAT_051', 'use_USRCAT_020', 'use_GETAWAY_029']
    return desc_list

def build_feat_map(dataset):
    if dataset == 'freesolv':
        # 임시 scgas용
        df3d = pd.read_csv(r'.\datasets\features_all1_drop_zeros.csv')
        df3d = df3d[selected_descriptors_scgas()]
    elif dataset == 'esol':
        # 임시 scgas용
        df3d = pd.read_csv(r'.\datasets\features_all1_drop_zeros.csv')
        df3d = df3d[selected_descriptors_scgas()]
    elif dataset == 'lipo':
        df3d = pd.read_csv(r'.\datasets\features_all1_drop_zeros.csv')
        df3d = df3d[selected_descriptors_scgas()]
    elif dataset == 'scgas':
        df3d = pd.read_csv(r'.\datasets\features_all1_drop_zeros.csv')
        df3d = df3d[selected_descriptors_scgas()]
    elif dataset == 'solubility':
        # 임시 scgas용
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
    # df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(fillna_value)

    # smiles -> np.array(D3,)
    feat_map = dict(zip(
        df[smiles_col].values,
        df[feature_cols].values.astype(np.float32)))

    return feat_map, feature_cols