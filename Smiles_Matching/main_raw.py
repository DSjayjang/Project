import numpy as np
import pandas as pd
from utils import sqrt_js_divergence

ms = pd.read_excel("./smiles_122.xlsx")
ms["m/z"] = ms["m/z"].astype(int)
ms["intensity"] = ms["intensity"].astype(float)

names = ms["Name"].drop_duplicates().reset_index(drop=True)
namesidx = {n: idx for idx, n in enumerate(names)}
ms["m/z_idx"] = (np.round(ms["m/z"]) - 1).astype(int)
ms["row_idx"] = ms["Name"].map(namesidx)

N = ms["row_idx"].max() + 1
D = ms["m/z_idx"].max() + 1

data = np.zeros((N, D))
grouped = ms.groupby(["row_idx", "m/z_idx"])["intensity"].max()

row_idx = grouped.index.get_level_values(0).to_numpy()
col_idx = grouped.index.get_level_values(1).to_numpy()
values = grouped.to_numpy()

data[row_idx, col_idx] = values

noise_set = [0.03, 0.05, 0.10]
case_set = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

avg_acc = np.zeros(len(noise_set))
avg_f1 = np.zeros(len(noise_set))

for i in range(len(noise_set)):
    
    accuracy = np.zeros(len(case_set))
    f1 = np.zeros(len(case_set))
    
    for j in range(len(case_set)):
        
        jsd_predict_acc = np.zeros(N)
        ter_true = np.zeros(N)
        ter_pred = np.zeros(N)
        
        for k in range(N):
            noise_data = data[k].copy()

            idx = data[k].argsort()
            peak2 = idx[-2]
            peak3 = idx[-3]
            noise_data[peak2] = data[k][peak2] * (1 + (noise_set[i] * case_set[j][0]))
            noise_data[peak3] = data[k][peak3] * (1 + (noise_set[i] * case_set[j][1]))
            
            noise_data /= noise_data.sum()
            
            jsd = np.zeros(N)
            
            for l in range(N):
                jsd[l] = sqrt_js_divergence(noise_data, data[l])
                
            pred = jsd.argmin()
        
            if pred == k:
                jsd_predict_acc[k] += 1
                         
        accuracy[j] = jsd_predict_acc.mean() * 100

    avg_acc[i] = accuracy.mean()

print("정확도: ", round(avg_acc.mean(), 2), "%", sep="")