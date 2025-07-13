import numpy as np

def FeatureNormalization(mol_graphs, feat_name):
    """
    feature(z-score) 정규화 함수
    mol_graphs 리스트의 각 그래프 객체에 대해 지정된 속성을 평균 0, 표준편차 1로 변환
    """
    features = [getattr(g, feat_name) for g in mol_graphs]
    features_mean = np.mean(features)
    features_std = np.std(features)

    for g in mol_graphs:
        val = getattr(g, feat_name)
        if features_std == 0:
            setattr(g, feat_name, 0)
        else:
            setattr(g, feat_name, (val - features_mean) / features_std)

def Z_Score(X):
    """
    z-score 표준화 함수
    1차원 또는 2차원 배열에 대해 평균 0, 표준편차 1로 변환
    표준편차가 0인 경우 0으로 처리
    """
    if len(X.shape) == 1:
        means = np.mean(X)
        stds = np.std(X)

        for i in range(0, X.shape[0]):
            if stds == 0:
                X[i] = 0
            else:
                X[i] = (X[i] - means) / stds
    else:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                if stds[j] == 0:
                    X[i, j] = 0
                else:
                    X[i, j] = (X[i, j] - means[j]) / stds[j]
    return X

def adj_mat_to_edges(adj_mat):
    """
    인접 행렬(adjacency matrix) 을 받아서,
    그 행렬에서 값이 1인 모든 위치 (i, j) 쌍을 모아 엣지 리스트(edge list) 형태로 반환
    """
    edges = []

    for i in range(0, adj_mat.shape[0]):
        for j in range(0, adj_mat.shape[1]):
            if adj_mat[i, j] == 1:
                edges.append((i, j))

    return edges

def atoms_to_symbols(atoms):
    # symbols = []

    # for atom in atoms:
    #     symbols.append(atom.GetSymbol())

    # return symbols
    return [atom.GetSymbol() for atom in atoms]