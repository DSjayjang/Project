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

meta = (ms[["row_idx", "Name"] + (["SMILES"] if "SMILES" in ms.columns else [])]
        .drop_duplicates(subset=["row_idx"])
        .sort_values("row_idx")
        .reset_index(drop=True))

noise_set = [0.03, 0.05, 0.10]
case_set = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

noise = noise_set[2]
case = case_set[1]

avg_acc = np.zeros(len(noise_set))


for k in range(N):
    noise_data = data[k].copy()
    
    idx = data[k].argsort()
    peak2 = idx[-2]
    peak3 = idx[-3]
    noise_data[peak2] = data[k][peak2] * (1 + (noise * case[0]))
    noise_data[peak3] = data[k][peak3] * (1 + (noise * case[1]))
    
    noise_data /= noise_data.sum()
    
    jsd = np.zeros(N)
    
    for l in range(N):
        jsd[l] = sqrt_js_divergence(noise_data, data[l])
        
    pred = jsd.argmin()

    name_k = meta.loc[k, "Name"]
    name_p = meta.loc[pred, "Name"]

    if "SMILES" in meta.columns:
        smi_k = meta.loc[k, "SMILES"]
        smi_p = meta.loc[pred, "SMILES"]
        print(f"target: {smi_k} -> pred={pred}: {name_p} ({smi_p})")
    else:
        print(f"WRONG PRED: target: {name_k} -> pred={pred}: {name_p}")