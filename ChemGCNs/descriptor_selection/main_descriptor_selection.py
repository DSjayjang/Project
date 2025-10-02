import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import SET_SEED, SEED, DATASET_PATH
from utils import MolecularFeatureExtractor

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



# nunique == 1 인 경우는 제
unique_columns = list(df_removed_features.loc[:, df_removed_features.nunique() == 1].columns)
print('nunique == 1인 feature : \n', unique_columns, '\n')

# nunique == 1인 feature 제거
#df_removed_features.drop(columns = unique_columns, inplace = True)
df_removed_features = df_removed_features.drop(columns = unique_columns).copy()

num_removed_features = df_removed_features.shape[1] - 1  # logvp 열 제외

print("제거 후 남은 feature 개수:", num_removed_features, '\n')
print(df_removed_features.shape)


# 너무 낮은 vairnace를 가지는 경
low_variances = sorted(df_removed_features.var())
low_variances[:10]

columns_low_variances = []

for i in low_variances:
    if i < 0.001:
        column = df_removed_features.loc[:, df_removed_features.var() == i].columns
        columns_low_variances.append(column)
columns_low_variances = [item for index in columns_low_variances for item in index]

# 2. 중복 제거 및 유니크 값 추출
columns_low_variances = list(set(columns_low_variances))
print(columns_low_variances)

# 낮은 분산의 변수 제거
df_removed_features = df_removed_features.drop(columns = columns_low_variances).copy()
num_removed_features = df_removed_features.shape[1] - 1  # logvp 열 제외

print("제거 후 남은 feature 개수:", num_removed_features, '\n')
print(df_removed_features.shape)



# ISIS
from isis import ISIS
screening = ISIS(df_removed_features)
selected_feats = screening.fit()
df_screening = screening.transform()

print(selected_feats, df_screening)


# elastic net
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

X_ISIS = df_screening.drop(columns = ['target'])
y_ISIS = df_screening['target']


# train / test split
X_train, X_test, y_train, y_test = train_test_split(X_ISIS, y_ISIS, test_size = 0.2, random_state = SEED)

# scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaling = scaler.transform(X_train)
X_test_scaling = scaler.transform(X_test)

iter = 5000

# ElasticNet 모델과 하이퍼파라미터 범위 설정
en = ElasticNet(max_iter = iter)

param_grid = {
    'alpha': np.linspace(0.01, 1.0, 300),  # 정규화 강도
    'l1_ratio': np.linspace(0.1, 0.9, 30)  # L1과 L2 비율
}

kfold = KFold(n_splits = 5, shuffle = True, random_state = SEED)

# GridSearchCV를 사용하여 최적 하이퍼파라미터 탐색
grid_search = GridSearchCV(
    estimator = en,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv = kfold,
    verbose = 0,
    n_jobs = -1
)
grid_search.fit(X_train_scaling, y_train)

best_params = grid_search.best_params_

best_en = ElasticNet(
    alpha = best_params['alpha'],
    l1_ratio = best_params['l1_ratio'],
    max_iter = iter,
    fit_intercept=True
)

# 적합
best_en.fit(X_train_scaling, y_train)

coefficients = best_en.coef_
coefficients.size

# 엘라스틱넷 적합이후 모든 변수
selected_features_elastic = list(X_train.loc[:, best_en.coef_ != 0].columns)

print(f'# {len(X_train.loc[:, best_en.coef_ != 0].columns)}개')
print(f'df_all =', selected_features_elastic, '\n')

# 계수 0인 변수
print(f'회귀계수가 0 인 변수: {len(X_train.loc[:, best_en.coef_ == 0].columns)}개')
print(f'df_all =', X_train.columns[best_en.coef_ == 0], '\n')



from sklearn.metrics import mean_squared_error, r2_score

y_pred = best_en.predict(X_test_scaling)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Test  MSE : {mse:.4f}")
print(f"Test   R² : {r2:.4f}")
n_selected = np.sum(best_en.coef_ != 0)
print(f"Selected features: {n_selected} / {X_train.shape[1]}")

import matplotlib.pyplot as plt

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 출력용
df_screening.describe()


# 계수가 0인 변수는 제거
X_train_None0 = X_train.drop(columns=X_train.columns[best_en.coef_ == 0])
coefficients_None0 = coefficients[coefficients!=0]

# 특성과 회귀계수 매핑
df_final_selected_features = pd.DataFrame({'Feature' : X_train_None0.columns,
                                       'Coefficient' : coefficients_None0})

# 계수 큰 값 기준으로
final_selected_features = abs(df_final_selected_features['Coefficient']).sort_values(ascending = False)
final_selected_features_index = final_selected_features.index

X_train_None0 = X_train_None0.iloc[:, final_selected_features_index]




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

    for i in range(len(elastic_list_copy)):
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

    for i in range(len(elastic_list_copy)):
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

    for i in range(len(elastic_list_copy)):
        if n % 5 == 0: print(f'# {n+1}')
        print(f"self_feats[i, {n}] = mol_graph.{X_train_None0_copy.columns[i]}")
        idx += 1
        n += 1

# 출력
mol_conv_upper(selected_features_elastic, X_train_None0)
mol_conv_under(selected_features_elastic, X_train_None0)
exec_reg(selected_features_elastic, X_train_None0)
print(X_train_None0.columns)