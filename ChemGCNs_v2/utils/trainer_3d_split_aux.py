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

from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K, SEED

lambda_desc = 0.001

def train(model, criterion, optimizer, train_loader, val_loader, max_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(0, max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for bg, self_feat, x3d, target in train_loader:
            pred, aux = model(bg, self_feat, x3d, return_aux = True)

            y_loss = criterion(pred, target)
            desc_loss = aux["nll_2d"].mean()  # batch 평균
            loss = y_loss + lambda_desc * desc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = target.size(0)
            train_loss_sum += y_loss.detach().item() * bs
            train_n += bs
        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for bg, self_feat, x3d, target in val_loader:
                pred, aux = model(bg, self_feat, x3d, return_aux = True)

                y_loss = criterion(pred, target)

                bs = target.size(0)
                val_loss_sum += y_loss.detach().item() * bs
                val_n += bs
        val_loss = val_loss_sum / val_n

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1} | train loss {train_loss:.4f} | val loss {val_loss:.4f}')
    
    return train_losses, val_losses


def test(model, criterion, test_data_loader):
    model.eval()

    all_preds = []
    all_targets = []
    all_self_feats = []

    with torch.no_grad():
        loss_sum = 0.0
        n_total = 0
        test_loss = 0
        desc_sum = 0.0    # optional: log only
        total_sum = 0.0   # optional: log only

        for bg, self_feat, x3d, target in test_data_loader:
            pred, aux = model(bg, self_feat, x3d, return_aux = True)

            y_loss = criterion(pred, target)

            bs = target.size(0)
            loss_sum +=y_loss.detach().item() * bs
            n_total += bs

            # aux loss
            if aux is not None and 'nll_2d' in aux:
                dloss = aux['nll_2d'].mean().item()
            else:
                dloss = 0.0
            desc_sum += dloss * bs

            # total loss
            total_sum += (y_loss.item() + lambda_desc * dloss) * bs

            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())
            all_self_feats.append(self_feat.detach().cpu())

        test_loss = loss_sum / n_total
        test_desc = desc_sum / n_total
        test_total = total_sum / n_total

        print(f"Test Loss: {test_loss}")
        print(f"Test custom(nll_2d) (log only): {test_desc}")
        print(f"Test total (loss + labmda_desc*custom) (reference): {test_total}")

    final_targets = torch.cat(all_targets, dim=0).numpy()
    final_preds = torch.cat(all_preds, dim=0).numpy()
    final_self_feats = torch.cat(all_self_feats, dim=0).numpy()

    # ✅ R^2 계산 (shape 안전 처리)
    y_true = np.squeeze(final_targets)  # (N,) 또는 (N,1) -> (N,)
    y_pred = np.squeeze(final_preds)
    r2 = r2_score(y_true, y_pred)
    print(f"Test R^2: {r2}")

    np.savetxt(r'.\results\result.csv', np.concatenate((final_targets, final_preds, final_self_feats), axis=1), delimiter=',')

    return test_loss, final_preds, r2

def cross_validation(dataset, model, criterion, num_folds, batch_size, max_epochs, collate, model_name):
    num_data_points = len(dataset)
    # size_fold = int(len(dataset) / float(num_folds))
    # folds = []
    models = []
    optimizers = []
    # test_losses = []
    LR = 3e-4

    fold_train_losses = []
    fold_val_losses = []
    fold_test_losses = []
    fold_test_r2 = []

    idx = np.arange(num_data_points)
    train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=SEED, shuffle=True)
    
    for k in range(0, num_folds):
        models.append(copy.deepcopy(model))
        optimizers.append(optim.Adam(models[k].parameters(), weight_decay=0.01, lr = LR))

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    # k-fold CV
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)

    for fold, (tr_sub, val_sub) in enumerate(kf.split(train_idx)):
        print(f'--------------- Fold {fold+1} ---------------')

        fold_train_idx = train_idx[tr_sub]
        fold_val_idx = train_idx[val_sub]
        
        train_loader = DataLoader(
            Subset(dataset, fold_train_idx),
            batch_size=batch_size, shuffle=True, collate_fn=collate
        )

        val_loader = DataLoader(
            Subset(dataset, fold_val_idx),
            batch_size=batch_size, shuffle=False, collate_fn=collate)
        
        train_losses, val_losses = train(models[fold], criterion, optimizers[fold], train_loader, val_loader, max_epochs)

        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

        test_loss, pred, r2 = test(models[fold], criterion, test_loader)
        fold_test_losses.append(test_loss)
        fold_test_r2.append(r2)

    # Plot fold별 loss
    save_path = f'./results/loss/{DATASET_NAME}_{model_name}_{MAX_EPOCHS}_{SEED}_{criterion}.png'
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
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    df_row = pd.DataFrame([{
        'dataset': DATASET_NAME,
        'model_name': model_name,
        'epochs': MAX_EPOCHS,
        'num-folds': K,
        'SEED': SEED,
        'criterion': criterion,
        'fold_test_losses': fold_test_losses,
        'fold_test_r2': fold_test_r2,
    }])
    csv_path = f'./results/se/{DATASET_NAME}_{model_name}_{MAX_EPOCHS}_{K}_{SEED}_{criterion}.csv'

    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode='a', header=False, index=False)

    return np.mean(fold_test_losses), np.mean(fold_test_r2)
