import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from model import test0219_grid, test0220_grid, test0221_grid

def _lap_pe_sign_flip_inplace(bg):
    """
    In-place sign flip for Laplacian PE (eigenvector sign ambiguity).
    bg.ndata['lap_pos_enc']: (N_total, k)
    """
    if 'lap_pos_enc' not in bg.ndata:
        return

    lap = bg.ndata['lap_pos_enc']
    # lap: (N_total, k)
    k = lap.size(1)

    # sign per eigen-dimension (k,), shared across all nodes in the batch graph
    sign_flip = torch.rand(k, device=lap.device)
    sign_flip = torch.where(sign_flip >= 0.5, 1.0, -1.0).to(lap.dtype)

    bg.ndata['lap_pos_enc'] = lap * sign_flip.unsqueeze(0)  # (N_total,k) * (1,k)

def train(model, criterion, optimizer, train_loader, val_loader, max_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(0, max_epochs):
        train_loss_sum = 0.0
        val_loss_sum = 0.0
        train_n = 0
        val_n = 0
        # ------------------ Train ------------------ #
        model.train()
        for bg, feat_2d, feat_3d, target in train_loader:
            _lap_pe_sign_flip_inplace(bg)
            pred = model(bg, feat_2d, feat_3d)

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = target.size(0)
            train_loss_sum += loss.detach().item() * bs
            train_n += bs
        train_loss = train_loss_sum / train_n

        # ------------------ Val ------------------ #
        model.eval()
        with torch.no_grad():
            for bg, feat_2d, feat_3d, target in val_loader:
                pred = model(bg, feat_2d, feat_3d)

                loss = criterion(pred, target)

                bs = target.size(0)
                val_loss_sum += loss.detach().item() * bs
                val_n += bs

        val_loss = val_loss_sum / val_n

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    return train_losses, val_losses


def grid_search_kfold(dataset, dim_in, dim_2d_desc, dim_3d_desc,
                      param_list, criterion, num_folds, batch_size, max_epochs, seed, dataset_name, collate_fn, model_name):

    device="cuda"

    lr = 1e-3
    weight_decay = 0.01

    num_data = len(dataset)
    idx = np.arange(num_data)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    results = []

    for cfg_id, cfg in enumerate(param_list):
        fold_scores = []
        fold_val_losses = []
        fold_train_losses = []

        print(f"\n================ CFG {cfg_id+1}/{len(param_list)} ================")
        print("cfg:", cfg)

        for fold, (train_idx, val_idx) in enumerate(kf.split(idx)):
            print(f"--------------- Fold {fold+1}/{num_folds} ---------------")

            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=batch_size, shuffle=True,
                collate_fn=collate_fn, drop_last=False)
            
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, drop_last=False)

            # model = test0220_grid.Net(
            #     dim_in=dim_in,
            #     dim_2d_desc=dim_2d_desc,
            #     dim_3d_desc=dim_3d_desc,
            #     d_t=cfg["d_t"],
            #     d_k=cfg["d_k"],
            #     d_h=cfg["d_h"],
            #     dim_out_fc1=cfg["fc1"],
            #     dim_out_fc2=cfg["fc2"],
            #     drop_out=cfg["dropout"],
            #     num_heads=cfg["num_heads"],
            #     rank=cfg["rank"],
            # ).to(device)
            model = test0221_grid.Net_total(
                dim_in=dim_in,
                dim_2d_desc=dim_2d_desc,
                dim_3d_desc=dim_3d_desc,
                d_h=cfg["d_h"],
                dim_out_fc1=cfg["fc1"],
                dim_out_fc2=cfg["fc2"],
                drop_out=cfg["dropout"],
                rank=cfg["rank"],
            ).to(device)


            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            train_losses, val_losses = train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=max_epochs)

            fold_train_losses.append(train_losses)
            fold_val_losses.append(val_losses)

            fold_score = float(np.mean(val_losses)) # mean not min
            fold_scores.append(fold_score)

            print(f"[Fold {fold+1}] mean val loss: {fold_score:.4f}")

        mean_loss = float(np.mean(fold_scores))

        results.append({
            "cfg": cfg,
            "mean_loss": mean_loss,
            "fold_scores": fold_scores,
            "fold_train_losses": fold_train_losses,
            "fold_val_losses": fold_val_losses,})

        print(f"\n[CFG {cfg_id+1}] CV(min val loss) = {mean_loss:.4f}")
        print("==========================================================")

        # ----------------------- save loss plot and csv file -----------------------#
        # loss plot
        save_path = f'./results/loss/{dataset_name}_{model_name}_{max_epochs}_{num_folds}_{seed}_{criterion}_{cfg["d_h"]}_{cfg["fc1"]}_{cfg["fc2"]}_{cfg["dropout"]}_{cfg["rank"]}.png'
        fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5), sharey=True)
        for k in range(num_folds):
            epochs = list(range(1, max_epochs + 1))
            axes[k].plot(epochs, fold_train_losses[k], label="Train Loss")
            axes[k].plot(epochs, fold_val_losses[k], label="Val Loss")
            axes[k].set_xlabel("Epoch")
            axes[k].set_ylabel("Loss")
            axes[k].legend()
            axes[k].set_title(f"Fold {k+1}")
        plt.suptitle("Train and Val Losses Across Folds")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # csv file
        df_row = pd.DataFrame([{
            'dataset': dataset_name,
            'model_name': model_name,
            'epochs': max_epochs,
            'num-folds': num_folds,
            'SEED': seed,
            'criterion': criterion,
            'fold_val_means': fold_scores}])
        csv_path = f'./results/se/{dataset_name}_{model_name}_{max_epochs}_{num_folds}_{seed}_{criterion}_{cfg["d_h"]}_{cfg["fc1"]}_{cfg["fc2"]}_{cfg["dropout"]}_{cfg["rank"]}.csv'
        if not os.path.exists(csv_path): df_row.to_csv(csv_path, index=False)
        else: df_row.to_csv(csv_path, mode='a', header=False, index=False)
        # ----------------------------------------------------------------------------- #

    results.sort(key=lambda x: x["mean_loss"])
    best = results[0]

    print("\n================ BEST params ================")
    print("Best cfg:", best["cfg"])
    print(f"CV score (mean of folds): {best['mean_loss']:.4f}")
    print("Fold means:", [f"{x:.6f}" for x in best["fold_scores"]])
    print("===============================================\n")

    return results
