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


def train(model, criterion, optimizer, train_loader, max_epochs, dataset_name, save_model):
    for epoch in range(max_epochs):
        train_loss_sum = 0.0
        train_n = 0

        model.train()
        # for bg, feat_2d, target, scaffold_id, _ in train_loader:
        #     pred, _, scaffold_logits = model(bg, feat_2d) # 
        #     loss = criterion(pred, target)
            
        #     if scaffold_logits is not None:
        #         loss_scaf = F.cross_entropy(scaffold_logits, scaffold_id) #
        #         lambda_scaf = 0.05
        #         loss = loss + lambda_scaf * loss_scaf

        for bg, feat_2d, target, scaffold_ids, _ in train_loader:
            outputs = model(bg, feat_2d)
            # print("target shape:", target.shape)
            # print("scaffold_ids shape:", scaffold_ids.shape)
            # print("scaffold_ids dtype:", scaffold_ids.dtype)
            # print("scaffold_ids[:10]:", scaffold_ids[:10])

            # print("scaffold_ids min/max:", scaffold_ids.min().item(), scaffold_ids.max().item())

            # print("aux logits shape:", outputs["scaffold_logits_aux"].shape)
            # print("adv logits shape:", outputs["scaffold_logits_adv"].shape)

            # print("aux out_features check:", outputs["scaffold_logits_aux"].size(1))
            # print("adv out_features check:", outputs["scaffold_logits_adv"].size(1))

            # raise RuntimeError("debug stop")
            pred = outputs["pred"]
            h_scaf = outputs["h_scaf"]
            h_phys = outputs["h_phys"]
            scaffold_logits_aux = outputs["scaffold_logits_aux"]
            scaffold_logits_adv = outputs["scaffold_logits_adv"]
            attn = outputs["attn"]
            loss, loss_dict = compute_total_loss(
                outputs=outputs,
                y=target,
                scaffold_ids=scaffold_ids,
                criterion_task=criterion,
                lambda_orth=0.05,
                lambda_aux=0.1,
                lambda_adv=0.1,
                lambda_cons=0.0,
            )


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = target.size(0)
            train_loss_sum += loss.detach().item() * bs
            train_n += bs
        train_loss = train_loss_sum / train_n

        print(f'[TRAIN] Epoch {epoch+1} | loss {train_loss:.4f}')

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





def test(model, criterion, test_loader, dataset_name, desc_list, mode=False):
    model.eval()
    loss_sum = 0.0
    test_n = 0

    preds, targets = [], []
    smiles_all = []
    all_attns=[]
    
    with torch.no_grad():
        for bg, feat_2d, target, scaffold_ids, _ in test_loader:
            outputs = model(bg, feat_2d)
            pred = outputs["pred"]
            h_scaf = outputs["h_scaf"]
            h_phys = outputs["h_phys"]
            scaffold_logits_aux = outputs["scaffold_logits_aux"]
            scaffold_logits_adv = outputs["scaffold_logits_adv"]
            attn = outputs["attn"]
            loss, loss_dict = compute_total_loss(
                outputs=outputs,
                y=target,
                scaffold_ids=scaffold_ids,
                criterion_task=criterion,
                lambda_orth=0.05,
                lambda_aux=0.1,
                lambda_adv=0.1,
                lambda_cons=0.0,
            )

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