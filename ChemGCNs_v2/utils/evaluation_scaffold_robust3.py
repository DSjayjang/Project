import time
import copy
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import torch.optim as optim

from utils.utils import plot_descriptor_importance
import torch.nn.functional as F

def train(model, criterion, optimizer, train_loader, max_epochs, dataset_name, save_model):
    total_time = 0
    
    lambda_orth=0.10
    lambda_aux=0.50
    lambda_cons=0.05
    
    for epoch in range(max_epochs):
        train_loss_sum = 0.0
        train_n = 0

        model.train()
        logs = {'task': 0., 'orth': 0., 'aux': 0., 'cons': 0., 'total': 0.}
        n = 0
        start_time = time.time() # 시작
        for bg, feat_2d, target, scaffold_ids, _ in train_loader:
            out, attn_list, h_scaf, h_phys  = model(bg, feat_2d) # 
            # ① 주 task 손실
            # task_loss = F.mse_loss(out.squeeze(-1), target)
            task_loss = criterion(out, target)

            if model.use_scaffold:
                # ② Scaffold-robustness 손실군
                sc_losses = model.compute_scaffold_losses(
                    h_scaf, h_phys, attn_list, scaffold_ids,
                    lambda_orth=lambda_orth,
                    lambda_aux=lambda_aux,
                    lambda_cons=lambda_cons,
                )

                loss = task_loss + sc_losses['total']
            else:
                sc_losses = {'orth': 0.0, 'aux': 0.0, 'cons': 0.0, 'total': 0.0}
                loss = task_loss
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # bs = target.size(0)
            # train_loss_sum += loss.detach().item() * bs
            # train_n += bs
            bs = target.size(0)
            if model.use_scaffold:
                logs['task']  += task_loss.item()       * bs
                logs['orth']  += sc_losses['orth'].item() * bs
                logs['aux']   += sc_losses['aux'].item()  * bs
                logs['cons']  += sc_losses['cons'].item() * bs
                logs['total'] += loss.item()      * bs
            else:
                train_loss_sum += loss.detach().item() * bs
            train_n += bs
        if model.use_scaffold:
            train_task = logs['task'] / train_n
            train_orth = logs['orth'] / train_n
            train_aux = logs['aux'] / train_n
            train_cons = logs['cons'] / train_n
            train_loss = logs['total'] / train_n
        else:
            train_loss = train_loss_sum / train_n
        epoch_time = time.time() - start_time # 끝
        total_time +=epoch_time
        if model.use_scaffold:
            print(f'[TRAIN] Epoch {epoch+1} | loss {train_loss:.4f} | time {epoch_time:.2f}s | train_task {train_task:.2f} | train_orth {train_orth} | train_aux {train_aux} | train_cons {train_cons}')
        else:
            print(f'[TRAIN] Epoch {epoch+1} | loss {train_loss:.4f} | time {epoch_time:.2f}s')

        # Save model
        if save_model:
            if (epoch + 1) == max_epochs:
                ckpt_dir = Path(f'./checkpoints/{dataset_name}')
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_path = ckpt_dir / f'model_{dataset_name}_{max_epochs}.pt'

                state_dict = model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

                torch.save(weights, save_path)
                print(f"{dataset_name} Model saved for {max_epochs}")

    avg_epoch_time = total_time / max_epochs
    return avg_epoch_time

def test(model, criterion, test_loader, dataset_name, desc_list, mode=False):
    model.eval()
    loss_sum = 0.0
    test_n = 0

    preds, targets = [], []
    smiles_all = []
    all_attns=[]
    
    with torch.no_grad():
        for bg, feat_2d, target, scaffold_ids, smiles in test_loader:
            pred, attn_list, h_scaf, h_phys = model(bg, feat_2d)

            # mean_attn = torch.stack(attn_list).mean(dim=0) # (bs, M)
            # all_attns.append(mean_attn.cpu())
            
            loss = criterion(pred, target)

            bs = target.size(0)
            loss_sum += loss.detach().item() * bs
            test_n += bs

            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
            # smiles_all.extend(smiles)

    test_loss = loss_sum / test_n

    y_true = torch.cat(targets, dim=0).numpy().squeeze()
    y_pred = torch.cat(preds, dim=0).numpy().squeeze()
    test_r2 = float(r2_score(y_true, y_pred))

    print(f'[TEST] loss {test_loss:.4f}')

    targets_np = torch.cat(targets, dim=0).cpu().numpy().reshape(-1)
    preds_np = torch.cat(preds, dim=0).cpu().numpy().reshape(-1)

    if mode:
        df = pd.DataFrame({
            'smiles': smiles_all,
            'target': targets_np,
            'pred': preds_np
        })
        df.to_csv(rf'./results/{dataset_name}.csv', index=False)

    # # 시각화    
    # total_avg_attn = torch.cat(all_attns, dim=0).mean(dim=0).numpy()
    # plot_descriptor_importance(total_avg_attn, desc_list)
    
    return test_loss, test_r2


def evaluation(train_dataset, test_dataset, model, criterion, desc_list, batch_size, max_epochs, collate_fn, dataset_name, phase, save_model, ckpt_path, model_name, lr=1e-3, weight_decay=0.001):
    m = copy.deepcopy(model)
    # m = model
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if phase=='train':
        opt = optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
        avg_epoch_time = train(m, criterion, opt, train_loader, max_epochs, dataset_name, save_model)

        print(f"\n epoch 평균 시간: {avg_epoch_time:.4f} 초\n")

        test_loss, test_r2 = test(m, criterion, test_loader, dataset_name, desc_list, mode=False)

        return test_loss, test_r2
    
    elif phase=='test':
        weights = torch.load(ckpt_path)
        m.load_state_dict(weights, strict=True)
        print(f"[LOAD] Loaded weights from: {ckpt_path}")
        
        test_loss, test_r2 = test(m, criterion, test_loader, dataset_name, desc_list, mode=True)
    
        return test_loss, test_r2