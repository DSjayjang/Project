import torch
import copy
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

import os
import pandas as pd
import matplotlib.pyplot as plt

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

        print(f'Epoch {epoch + 1} | train loss {train_loss:.4f} | val loss {val_loss:.4f}')

    return train_losses, val_losses


def cross_validation(dataset, model, criterion, num_folds, batch_size, max_epochs, seed, dataset_name, collate, model_name, lr=1e-3, weight_decay = 0.01):
    num_data = len(dataset)
    fold_train_losses = []
    fold_val_losses = []
    fold_val_scores = []

    idx = np.arange(num_data)

    # K-Fold CV
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(idx)):
        print(f'--------------- Fold {fold+1} ---------------')

        m = copy.deepcopy(model)
        opt = optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size, shuffle=True, collate_fn=collate)
        
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size, shuffle=False, collate_fn=collate)

        train_losses, val_losses = train(m, criterion, opt, train_loader, val_loader, max_epochs)
        
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

        fold_score = float(np.mean(val_losses)) # np.min으로 하기
        fold_val_scores.append(fold_score)

        print(f"[Fold {fold+1}] Val loss: {fold_score:.4f}")

    cv_mean_val = float(np.mean(fold_val_scores))

    # ----------------------- save loss plot and csv file -----------------------#
    # loss plot
    save_path = f'./results/loss/{dataset_name}_{model_name}_{max_epochs}_{num_folds}_{seed}_{criterion}.png'
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
        'fold_val_means': fold_val_scores}])
    csv_path = f'./results/se/{dataset_name}_{model_name}_{max_epochs}_{num_folds}_{seed}_{criterion}.csv'
    if not os.path.exists(csv_path): df_row.to_csv(csv_path, index=False)
    else: df_row.to_csv(csv_path, mode='a', header=False, index=False)
    # ----------------------------------------------------------------------------- #

    return cv_mean_val

    #     fold_train_idx = train_idx[tr_sub]
    #     fold_val_idx = val_idx[val_sub]
        
    #     train_loader = DataLoader(
    #         Subset(dataset, fold_train_idx),
    #         batch_size=batch_size, shuffle=True, collate_fn=collate
    #     )

    #     val_loader = DataLoader(
    #         Subset(dataset, fold_val_idx),
    #         batch_size=batch_size, shuffle=False, collate_fn=collate)
        
    #     m = copy.deepcopy(model)
    #     opt = optim.Adam(m.parameters(), weight_decay=weight_decay)

    #     train_loader = DataLoader(Subset(dataset, train_idx))


    #     train_losses, val_losses = train(m, criterion, opt, train_loader, val_loader, max_epochs)

    #     fold_train_losses.append(train_losses)
    #     fold_val_losses.append(val_losses)

    # # Test
    # final_model = copy.deepcopy(model)
    # final_opt = optim.Adam(final_model.parameters(), weight_decay=0.01, lr=LR)

    # final_train_loader = DataLoader(Subset())

    # test_loss, pred, r2 = test(models[fold], criterion, test_loader)
    # fold_test_losses.append(test_loss)
    # fold_test_r2.append(r2)



    # return np.mean(fold_test_losses), np.mean(fold_test_r2)
