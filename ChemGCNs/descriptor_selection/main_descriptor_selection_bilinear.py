import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import SET_SEED, SEED, DATASET_PATH
from descriptor_selection.utils import MolecularFeatureExtractor

# 재현성 난수 고정
SET_SEED()

df = pd.read_csv(DATASET_PATH + '.csv')
smiles_list = df['smiles'].tolist()

# target 정의
target = df.iloc[:,-1]
print(smiles_list[:5])
print(target[:5])



# 분자 특성 추출 및 데이터프레임 정의
extractor = MolecularFeatureExtractor()
df_all_features = extractor.extract_molecular_features(smiles_list)

df_all_features['target'] = target
df_all_features

num_all_features = df_all_features.shape[1] - 1 
print("초기 변수 개수:", num_all_features)

# na handling
# NA 확인
df_all_features[df_all_features.isna().any(axis = 1)] # 행방향

# 결측치가 포함된 feature 개수
print('결측치가 포함된 열 개수:', df_all_features.isna().any(axis = 0).sum(), '\n')
print(df_all_features.isna().any(axis = 0))

print('결측치가 포함된 행 개수:', df_all_features.isna().any(axis = 1).sum(), '\n')
print(df_all_features.isna().any(axis = 1))

df_removed_features = df_all_features.dropna()

# 결측치가 포함된 feature 제거
# df_removed_features = df_all_features.dropna(axis = 1)
num_removed_features = df_removed_features.shape[1] - 1  # logvp 열 제외

print("제거 후 남은 feature 개수:", num_removed_features)

# 결측치가 제거된 data frame
df_removed_features

# 결측치가 포함된 feature 개수
print('결측치가 포함된 열 개수:', df_removed_features.isna().any(axis = 0).sum(), '\n')
print(df_removed_features.isna().any(axis = 0))

print('결측치가 포함된 행 개수:', df_removed_features.isna().any(axis = 1).sum(), '\n')
print(df_removed_features.isna().any(axis = 1))

print(df_removed_features)

# # nunique == 1 인 경우는 제
# unique_columns = list(df_removed_features.loc[:, df_removed_features.nunique() == 1].columns)
# print('nunique == 1인 feature : \n', unique_columns, '\n')

# # nunique == 1인 feature 제거
# #df_removed_features.drop(columns = unique_columns, inplace = True)
# df_removed_features = df_removed_features.drop(columns = unique_columns).copy()

# num_removed_features = df_removed_features.shape[1] - 1  # logvp 열 제외

# print("제거 후 남은 feature 개수:", num_removed_features, '\n')
# print(df_removed_features.shape)


# # 너무 낮은 vairnace를 가지는 경
# low_variances = sorted(df_removed_features.var())
# low_variances[:10]

# columns_low_variances = []

# for i in low_variances:
#     if i < 0.001:
#         column = df_removed_features.loc[:, df_removed_features.var() == i].columns
#         columns_low_variances.append(column)
# columns_low_variances = [item for index in columns_low_variances for item in index]

# # 2. 중복 제거 및 유니크 값 추출
# columns_low_variances = list(set(columns_low_variances))
# print(columns_low_variances)

# # 낮은 분산의 변수 제거
# df_removed_features = df_removed_features.drop(columns = columns_low_variances).copy()
# num_removed_features = df_removed_features.shape[1] - 1  # logvp 열 제외

# print("제거 후 남은 feature 개수:", num_removed_features, '\n')
# print(df_removed_features.shape)


"""
단지 출력용
나중에 지워도 됨
"""
def mol_conv_upper(elastic_list, df):
    print('mol_conv.py upper')

    elastic_list_copy = elastic_list.copy()
    X_train_None0_copy = df.copy()

    idx = 0
    n = 0

    for i in range(len(X_train_None0_copy.columns)):
        if n % 5 == 0: print(f'# {n+1}')
        print(f"mol_graph.{X_train_None0_copy.columns[i]} = dsc.{X_train_None0_copy.columns[i]}(mol)")
        idx += 1
        n += 1

def mol_conv_under(elastic_list, df):
    print('\n')
    print('mol_conv.py under')

    elastic_list_copy = elastic_list.copy()
    X_train_None0_copy = df.copy()

    idx = 0
    n = 0

    for i in range(len(X_train_None0_copy.columns)):
        if n % 5 == 0: print(f'# {n+1}')
        print(f"normalize_self_feat(mol_graphs, '{X_train_None0_copy.columns[i]}')")
        idx += 1
        n += 1

def exec_reg(elastic_list, df):
    print('\n')
    print('exec_reg.py')

    elastic_list_copy = elastic_list.copy()
    X_train_None0_copy = df.copy()

    idx = 0
    n = 0

    for i in range(len(X_train_None0_copy.columns)):
        if n % 5 == 0: print(f'# {n+1}')
        print(f"self_feats[i, {n}] = mol_graph.{X_train_None0_copy.columns[i]}")
        idx += 1
        n += 1

# 출력
mol_conv_upper(df_removed_features, df_removed_features)
mol_conv_under(df_removed_features, df_removed_features)
exec_reg(df_removed_features, df_removed_features)
# for i in range(len(df_removed_features.columns)):
#     print(df_removed_features.columns[i])
print(", ".join([f"'{col}'" for col in df_removed_features.columns]))
