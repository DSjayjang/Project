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
from model.BAN_robust import compute_total_loss
import math
import time
def attention_stability_loss(attn: torch.Tensor, scaffold_ids: torch.Tensor):
    """
    attn:         (bs, num_desc)
    scaffold_ids: (bs,)
    """
    unique_ids = torch.unique(scaffold_ids)
    global_mean = attn.mean(dim=0)

    loss = attn.new_tensor(0.0)
    count = 0

    for s in unique_ids:
        mask = (scaffold_ids == s)
        if mask.sum() >= 2:
            scaf_mean = attn[mask].mean(dim=0)
            loss = loss + F.mse_loss(scaf_mean, global_mean)
            count += 1

    if count == 0:
        return attn.new_tensor(0.0)

    return loss / count

def cross_scaffold_attention_consistency(attn: torch.Tensor,
                                         y: torch.Tensor,
                                         scaffold_ids: torch.Tensor,
                                         tau: float = 0.5):
    """
    attn:         (bs, num_desc)
    y:            (bs,) or (bs, 1)
    scaffold_ids: (bs,)
    """
    if y.dim() == 2 and y.size(1) == 1:
        y = y.squeeze(1)

    bs = attn.size(0)
    loss = attn.new_tensor(0.0)
    count = 0

    for i in range(bs):
        for j in range(i + 1, bs):
            # 다른 scaffold끼리만 비교
            if scaffold_ids[i] != scaffold_ids[j]:
                # target이 비슷할수록 가중치 크게
                w = torch.exp(-torch.abs(y[i] - y[j]) / tau)

                cos_sim = F.cosine_similarity(
                    attn[i].unsqueeze(0),
                    attn[j].unsqueeze(0),
                    dim=-1
                ).squeeze(0)

                loss = loss + w * (1.0 - cos_sim)
                count += 1

    if count == 0:
        return attn.new_tensor(0.0)

    return loss / count

def get_adv_lambda(epoch, max_epochs, max_lambda=0.2):
    p = epoch / max_epochs
    base = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
    return max_lambda * base



# def train(model, criterion, optimizer, train_loader, max_epochs, dataset_name, save_model):
#     for epoch in range(max_epochs):
#         train_loss_sum = 0.0
#         train_n = 0

#         model.train()
#         # for bg, feat_2d, target, scaffold_id, _ in train_loader:
#         #     pred, _, scaffold_logits = model(bg, feat_2d) # 
#         #     loss = criterion(pred, target)
            
#         #     if scaffold_logits is not None:
#         #         loss_scaf = F.cross_entropy(scaffold_logits, scaffold_id) #
#         #         lambda_scaf = 0.05
#         #         loss = loss + lambda_scaf * loss_scaf

#         for bg, feat_2d, target, scaffold_ids, _ in train_loader:
#             # outputs = model(bg, feat_2d)
#             # loss = criterion(outputs, target)
#             pred, attn_list, scaf_logits = model(bg, feat_2d)

#             loss_task = criterion(pred, target)
#             loss_scaf = F.cross_entropy(scaf_logits, scaffold_ids)

#             loss = loss_task + 1.0 * loss_scaf

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             bs = target.size(0)
#             train_loss_sum += loss.detach().item() * bs
#             train_n += bs
#         train_loss = train_loss_sum / train_n

#         print(f'[TRAIN] Epoch {epoch+1} | loss {train_loss:.4f}')

#         # Save model
#         if save_model:
#             if (epoch + 1) == max_epochs:
#                 ckpt_dir = Path(f'./checkpoints/{dataset_name}')
#                 ckpt_dir.mkdir(parents=True, exist_ok=True)
#                 save_path = ckpt_dir / f'model_{dataset_name}_{max_epochs}.pt'

#                 state_dict = model.state_dict()
#                 weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

#                 torch.save(weights, save_path)
#                 print(f"{dataset_name} Model saved for {max_epochs}")

def train(model,
          criterion,
          optimizer,
          train_loader,
          max_epochs,
           dataset_name, save_model,
          lambda_scaf=1.0,
          lambda_attn=0.1,
          use_pair_consistency=False,
          tau_y=0.5):
    
    for epoch in range(max_epochs):
        model.train()

        adv_lambda = get_adv_lambda(epoch, max_epochs, max_lambda=0.2)

        train_loss_sum = 0.0
        train_task_sum = 0.0
        train_adv_sum = 0.0
        train_attn_sum = 0.0
        train_n = 0

        start_time = time.time()

        for batch in train_loader:
            # 예시:
            # bg, feat_2d, target, scaffold_ids, smiles = batch
            bg, feat_2d, target, scaffold_ids, _ = batch


            pred, attn_list, scaf_logits = model(
                bg, feat_2d, adv_lambda=adv_lambda
            )

            # property loss
            loss_task = criterion(pred, target)

            # scaffold adversarial loss
            loss_adv = F.cross_entropy(scaf_logits, scaffold_ids)

            # attention stability loss
            attn_last = attn_list[-1]

            if use_pair_consistency:
                loss_attn = cross_scaffold_attention_consistency(
                    attn_last, target, scaffold_ids, tau=tau_y
                )
            else:
                loss_attn = attention_stability_loss(attn_last, scaffold_ids)

            # loss = loss_task + lambda_scaf * loss_adv + lambda_attn * loss_attn
            loss=loss_task
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = target.size(0)
            train_loss_sum += loss.detach().item() * bs
            train_task_sum += loss_task.detach().item() * bs
            train_adv_sum += loss_adv.detach().item() * bs
            train_attn_sum += loss_attn.detach().item() * bs
            train_n += bs

        epoch_time = time.time() - start_time

        print(
            f"[TRAIN] Epoch {epoch+1:03d} | "
            f"total {train_loss_sum/train_n:.4f} | "
            f"task {train_task_sum/train_n:.4f} | "
            f"adv {train_adv_sum/train_n:.4f} | "
            f"attn {train_attn_sum/train_n:.4f} | "
            f"adv_lambda {adv_lambda:.4f} | "
            f"time {epoch_time:.2f}s"
        )



def test(model, criterion, test_loader, dataset_name, desc_list, mode=False):
    model.eval()
    loss_sum = 0.0
    test_n = 0

    preds, targets = [], []
    smiles_all = []
    all_attns=[]
    
    with torch.no_grad():
        for bg, feat_2d, target, scaffold_ids, _ in test_loader:
            outputs, attn_list, scaf_logits  = model(bg, feat_2d)
            loss = criterion(outputs, target)

            bs = target.size(0)
            loss_sum += loss.detach().item() * bs
            test_n += bs

            preds.append(outputs.detach().cpu())
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
            # 'smiles': smiles_all,
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
        train(m, criterion, opt, train_loader, max_epochs, dataset_name, save_model)
        test_loss, test_r2 = test(m, criterion, test_loader, dataset_name, desc_list, mode=False)

        return test_loss, test_r2
    
    elif phase=='test':
        weights = torch.load(ckpt_path)
        m.load_state_dict(weights, strict=True)
        print(f"[LOAD] Loaded weights from: {ckpt_path}")
        
        test_loss, test_r2 = test(m, criterion, test_loader, dataset_name, desc_list, mode=True)
    
        return test_loss, test_r2