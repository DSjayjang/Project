import os
import time
import copy
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

from utils.utils import plot_descriptor_importance
import torch.nn.functional as F

def train(model, criterion, optimizer, train_loader, val_loader, max_epochs, dataset_name, save_model):
    train_losses = []
    val_losses = []
    total_time = 0.0
    
    best_val_loss = float('inf')
    best_epoch = -1
    best_state_dict = None

    for epoch in range(max_epochs):
        train_loss_sum = 0.0
        train_n = 0

        # train
        model.train()
        start_time = time.time()
        for bg, feat_2d, target, _ in train_loader:
            pred = model(bg, feat_2d) 
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = target.size(0)
            train_loss_sum += loss.detach().item() * bs
            train_n += bs

        train_loss = train_loss_sum / train_n
        epoch_time = time.time() - start_time
        total_time += epoch_time
        
        # validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for bg, feat_2d, target, _ in val_loader:
                pred = model(bg, feat_2d)
                loss = criterion(pred, target)

                bs = target.size(0)
                val_loss_sum += loss.detach().item() * bs
                val_n += bs

        val_loss = val_loss_sum / val_n

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1} | train loss {train_loss:.4f} | val loss {val_loss:.4f}')

        # best model = minimum val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())

        avg_epoch_time = total_time / max_epochs
        
        # # Save model
        # if save_model:
        #     if (epoch + 1) == max_epochs:
        #         ckpt_dir = Path(f'./checkpoints/{dataset_name}')
        #         ckpt_dir.mkdir(parents=True, exist_ok=True)
        #         save_path = ckpt_dir / f'model_{dataset_name}_{max_epochs}.pt'

        #         state_dict = model.state_dict()
        #         weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

        #         torch.save(weights, save_path)
        #         print(f"{dataset_name} Model saved for {max_epochs}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state_dict": best_state_dict,
        "avg_epoch_time": avg_epoch_time,
    }


def test(model, criterion, test_loader, dataset_name, desc_list, mode=False, save_path=None):
    model.eval()

    loss_sum = 0.0
    test_n = 0

    preds, targets = [], []
    smiles_all = []
    
    with torch.no_grad():
        for bg, feat_2d, target, smiles in test_loader:
            pred = model(bg, feat_2d)
            loss = criterion(pred, target)

            bs = target.size(0)
            loss_sum += loss.detach().item() * bs
            test_n += bs

            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
            smiles_all.extend(smiles)

    test_loss = loss_sum / test_n

    targets_np = torch.cat(targets, dim=0).cpu().numpy().reshape(-1)
    preds_np = torch.cat(preds, dim=0).cpu().numpy().reshape(-1)
    test_r2 = float(r2_score(targets_np, preds_np))

    print(f'[TEST] loss {test_loss:.4f}\n')

    if mode and save_path is not None:
        df = pd.DataFrame({
            'smiles': smiles_all,
            'target': targets_np,
            'pred': preds_np
        })
        df.to_csv(save_path, index=False)

    return {
        "test_loss": test_loss,
        "test_r2": test_r2,
        "preds": preds_np,
        "targets": targets_np,
        "smiles": smiles_all,
    }

def evaluation(dataset, model, criterion, desc_list, batch_size, max_epochs, seed, num_folds, collate_fn, dataset_name, phase, save_model, ckpt_path, model_name, lr=1e-3, weight_decay=0.01):
    num_data = len(dataset)
    idx = np.arange(num_data)

    fold_train_losses = []
    fold_val_losses = []

    fold_test_losses = []
    fold_test_r2s = []
    fold_avg_epoch_times = []

    os.makedirs('./results/loss', exist_ok=True)
    os.makedirs('./results/cv_pred', exist_ok=True)
    os.makedirs('./results/se', exist_ok=True)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for fold, (outer_train_idx, outer_test_idx) in enumerate(kf.split(idx)):
        print(f'--------------- Fold {fold+1} ---------------')

        # outer train -> inner train/val = 8:2
        inner_train_idx, inner_val_idx = train_test_split(
            outer_train_idx, test_size=0.2, shuffle=True, random_state=seed+fold
        )

        train_loader = DataLoader(
            Subset(dataset, inner_train_idx),
            batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        val_loader = DataLoader(
            Subset(dataset, inner_val_idx),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        test_loader = DataLoader(
            Subset(dataset, outer_test_idx),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        m = copy.deepcopy(model)


        opt = optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
        
        train_out = train(m, criterion, opt, train_loader, val_loader, max_epochs, dataset_name, save_model)

        fold_train_losses.append(train_out["train_losses"])
        fold_val_losses.append(train_out["val_losses"])
        fold_avg_epoch_times.append(train_out["avg_epoch_time"])

        # best epoch model load
        m.load_state_dict(train_out["best_state_dict"])

        # optional save model
        if save_model:
            ckpt_dir = Path(f'./checkpoints/{dataset_name}')
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_path = ckpt_dir / f'model_{dataset_name}_{model_name}_fold{fold+1}.pt'

            state_dict = m.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, save_path)
            print(f"{dataset_name} Fold {fold+1} model saved")

        pred_save_path = f'./results/cv_pred/{dataset_name}_{model_name}_{max_epochs}_{seed}_fold{fold+1}.csv'

        test_out = test(
            model=m,
            criterion=criterion,
            test_loader=test_loader,
            dataset_name=dataset_name,
            desc_list=desc_list,
            mode=True,
            save_path=pred_save_path
        )

        fold_test_losses.append(test_out["test_loss"])
        fold_test_r2s.append(test_out["test_r2"])

        print(
            f"[Fold {fold+1}] "
            f"best epoch: {train_out['best_epoch']} | "
            f"best val loss: {train_out['best_val_loss']:.4f} | "
            f"test loss: {test_out['test_loss']:.4f} | "
            f"test r2: {test_out['test_r2']:.4f}"
        )

    # 최적/최악 fold는 test loss 기준으로 비교
    # 이유: fold 간 최종 일반화 성능 비교이기 때문
    best_fold_idx = int(np.argmin(fold_test_losses))
    worst_fold_idx = int(np.argmax(fold_test_losses))

    # best fold
    best_loss = fold_test_losses[best_fold_idx]
    best_r2 = fold_test_r2s[best_fold_idx]

    # worst fold
    worst_loss = fold_test_losses[worst_fold_idx]
    worst_r2 = fold_test_r2s[worst_fold_idx]

    print(f"\n가장 loss가 낮았던 fold 번호: Fold {best_fold_idx + 1} -> loss: {best_loss:.4f}, R2: {best_r2:.4f} ")

    print(f"가장 loss가 높았던 fold 번호: Fold {worst_fold_idx + 1} -> loss: {worst_loss:.4f}, R2: {worst_r2:.4f}")

    cv_mean_test_loss = float(np.mean(fold_test_losses))
    cv_mean_test_r2 = float(np.mean(fold_test_r2s))
    cv_mean_epoch_time = float(np.mean(fold_avg_epoch_times))

    print(f"\n epoch 평균 시간: {cv_mean_epoch_time:.4f} 초\n")
    print(f"[CV] mean test loss: {cv_mean_test_loss:.4f}")
    print(f"[CV] mean test r2: {cv_mean_test_r2:.4f}")

    # ----------------------- save loss plot and csv file -----------------------#
    save_path = f'./results/loss/{dataset_name}_{model_name}_{max_epochs}_{num_folds}_{seed}_{criterion}.png'
    fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5), sharey=True)

    if num_folds == 1:
        axes = [axes]

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

    df_row = pd.DataFrame([{
        'dataset': dataset_name,
        'model_name': model_name,
        'epochs': max_epochs,
        'num-folds': num_folds,
        'SEED': seed,
        'criterion': criterion,
        'fold_test_losses': fold_test_losses,
        'fold_test_r2s': fold_test_r2s,
        'cv_mean_test_loss': cv_mean_test_loss,
        'cv_mean_test_r2': cv_mean_test_r2,
        'best_fold': best_fold_idx + 1,
        'worst_fold': worst_fold_idx + 1
    }])

    csv_path = f'./results/se/{dataset_name}_{model_name}_{max_epochs}_{num_folds}_{seed}_{criterion}.csv'
    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode='a', header=False, index=False)
    # ------------------------------------------------------------------------- #

    return cv_mean_test_loss, cv_mean_test_r2

        

        # test_loss, test_r2 = test(m, criterion, test_loader, dataset_name, desc_list, mode=True)

        # return test_loss, test_r2
    
        # elif phase=='test':
        #     weights = torch.load(ckpt_path)
        #     m.load_state_dict(weights, strict=True)
        #     print(f"[LOAD] Loaded weights from: {ckpt_path}")
            
        #     test_loss, test_r2 = test(m, criterion, test_loader, dataset_name, desc_list, mode=True)
        
        # return test_loss, test_r2