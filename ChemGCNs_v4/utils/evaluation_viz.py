import copy
import time
from pathlib import Path
from collections import OrderedDict

import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


def train(model, criterion, optimizer, train_loader, max_epochs, dataset_name, save_model):
    total_time = 0.0

    for epoch in range(max_epochs):
        train_loss_sum = 0.0
        train_n = 0

        model.train()
        start_time = time.time()

        for bg, feat_2d, target, _ in train_loader:
            pred, _, _ = model(bg, feat_2d)
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

        print(f'[TRAIN] Epoch {epoch+1} | loss {train_loss:.4f} | time {epoch_time:.2f}s')

        if save_model and (epoch + 1) == max_epochs:
            ckpt_dir = Path(f'./checkpoints/{dataset_name}')
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_path = ckpt_dir / f'model_{dataset_name}_{max_epochs}.pt'

            state_dict = model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, save_path)
            print(f"{dataset_name} Model saved for {max_epochs}")

    avg_epoch_time = total_time / max_epochs
    return avg_epoch_time


def test(
    model,criterion,test_loader,dataset_name,desc_list,mode=False,return_attention=False,save_attention_csv=False):
    model.eval()
    loss_sum = 0.0
    test_n = 0

    preds, targets = [], []
    smiles_all = []

    attention_records = []

    # model.family_order를 사용하는 것이 가장 안전
    family_order = model.family_order if hasattr(model, "family_order") else None

    with torch.no_grad():
        for bg, feat_2d, target, smiles in test_loader:
            pred, attn_dict, beta = model(bg, feat_2d)
            loss = criterion(pred, target)

            bs = target.size(0)
            loss_sum += loss.detach().item() * bs
            test_n += bs

            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
            smiles_all.extend(smiles)

            if return_attention:
                # 각 그래프의 실제 node 수
                num_nodes_list = bg.batch_num_nodes().tolist()

                if family_order is None:
                    family_order = list(attn_dict.keys())

                for i in range(bs):
                    trimmed_attn = {
                        fam: attn_dict[fam][i, :num_nodes_list[i]].detach().cpu().tolist()
                        for fam in family_order
                    }


                    # print("=" * 80)
                    # print("SAVE CHECK")
                    # print("smiles:", smiles[i])
                    # print("num_nodes:", num_nodes_list[i])
                    # for fam in family_order:
                    #     print(f"[{fam}] {trimmed_attn[fam]}")


                    beta_dict = {
                        fam: beta[i, j].detach().cpu().item()
                        for j, fam in enumerate(family_order)
                    }

                    attention_records.append({
                        "smiles": smiles[i],
                        "num_nodes": num_nodes_list[i],
                        "target": target[i].detach().cpu().reshape(-1)[0].item(),
                        "pred": pred[i].detach().cpu().reshape(-1)[0].item(),
                        "attn_dict": trimmed_attn,
                        "beta_dict": beta_dict,
                    })
                    # for i in range(bs):
                    #     print("=" * 80)
                    #     print("i:", i)
                    #     print("smiles:", smiles[i])
                    #     print("num_nodes:", num_nodes_list[i])

                    #     for fam in family_order:
                    #         alpha_i = attn_dict[fam][i, :num_nodes_list[i]].detach().cpu().tolist()
                    #         print(f"[{fam}] sum={sum(alpha_i):.6f}, len={len(alpha_i)}, alpha={alpha_i}")


    test_loss = loss_sum / test_n

    y_true = torch.cat(targets, dim=0).numpy().squeeze()
    y_pred = torch.cat(preds, dim=0).numpy().squeeze()
    test_r2 = float(r2_score(y_true, y_pred))

    print(f'[TEST] loss {test_loss:.4f} | R2 {test_r2:.4f}')

    targets_np = torch.cat(targets, dim=0).cpu().numpy().reshape(-1)
    preds_np = torch.cat(preds, dim=0).cpu().numpy().reshape(-1)

    if mode:
        results_dir = Path('./results')
        results_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            'smiles': smiles_all,
            'target': targets_np,
            'pred': preds_np
        })
        df.to_csv(results_dir / f'{dataset_name}.csv', index=False)

    if return_attention and save_attention_csv:
        results_dir = Path('./results')
        results_dir.mkdir(parents=True, exist_ok=True)

        # beta만 별도 csv로 저장
        beta_rows = []
        for rec in attention_records:
            row = {
                "smiles": rec["smiles"],
                "num_nodes": rec["num_nodes"],
                "target": rec["target"],
                "pred": rec["pred"],
            }
            for fam, val in rec["beta_dict"].items():
                row[f"beta_{fam}"] = val
            beta_rows.append(row)

        beta_df = pd.DataFrame(beta_rows)
        beta_df.to_csv(results_dir / f'{dataset_name}_beta.csv', index=False)

    if return_attention:
        return test_loss, test_r2, attention_records

    return test_loss, test_r2


def evaluation(
    train_dataset,test_dataset,model,criterion,desc_list,batch_size,max_epochs,collate_fn,dataset_name,phase,save_model,ckpt_path,model_name,lr=1e-3,weight_decay=0.01,return_attention=True,save_attention_csv=True,
):
    m = copy.deepcopy(model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    if phase == 'train':
        opt = optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
        avg_epoch_time = train(m, criterion, opt, train_loader, max_epochs, dataset_name, save_model)

        print(f"\n epoch 평균 시간: {avg_epoch_time:.4f} 초\n")

        if return_attention:
            test_loss, test_r2, attention_records = test(
                m,
                criterion,
                test_loader,
                dataset_name,
                desc_list,
                mode=True,
                return_attention=True,
                save_attention_csv=save_attention_csv,
            )
            return test_loss, test_r2, attention_records
        else:
            test_loss, test_r2 = test(
                m,
                criterion,
                test_loader,
                dataset_name,
                desc_list,
                mode=True,
                return_attention=False,
            )
            return test_loss, test_r2

    elif phase == 'test':
        weights = torch.load(ckpt_path)
        m.load_state_dict(weights, strict=True)
        print(f"[LOAD] Loaded weights from: {ckpt_path}")

        if return_attention:
            test_loss, test_r2, attention_records = test(
                m,
                criterion,
                test_loader,
                dataset_name,
                desc_list,
                mode=True,
                return_attention=True,
                save_attention_csv=save_attention_csv,
            )
            return test_loss, test_r2, attention_records
        else:
            test_loss, test_r2 = test(
                m,
                criterion,
                test_loader,
                dataset_name,
                desc_list,
                mode=True,
                return_attention=False,
            )
            return test_loss, test_r2