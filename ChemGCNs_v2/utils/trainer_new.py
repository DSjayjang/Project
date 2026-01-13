import torch
import copy
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import pandas as pd
from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K, SEED

def train_gcn(model, criterion, optimizer, train_data_loader, max_epochs):
    model.train()

    for epoch in range(0, max_epochs):
        train_loss = 0

        for bg, target in train_data_loader:
            pred = model(bg)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        train_loss /= len(train_data_loader.dataset)

        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))


def train_model(model, criterion, optimizer, train_data_loader, max_epochs):
    model.train()

    for epoch in range(0, max_epochs):
        train_loss = 0

        for bg, self_feat, x3d, target in train_data_loader:
            pred = model(bg, self_feat, x3d)

            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        train_loss /= len(train_data_loader.dataset)

        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))


# def train_model(model, criterion, optimizer, train_data_loader, max_epochs,
#                 log_attn_every_epoch=1, log_attn_every_batch=None, device=None):
#     """
#     log_attn_every_epoch: 몇 epoch마다 attention 로그 출력할지 (기본 1)
#     log_attn_every_batch: (선택) 몇 batch마다 attention 로그 출력할지. None이면 epoch 기준만 사용.
#     device: (선택) 텐서를 올릴 디바이스. 너 코드가 이미 내부에서 cuda 처리한다면 생략 가능.
#     """
#     model.train()

#     for epoch in range(max_epochs):
#         train_loss = 0.0

#         for batch_idx, (bg, self_feat, x3d, target) in enumerate(train_data_loader):

#             # # (선택) device로 이동
#             # if device is not None:
#             #     bg = bg.to(device)
#             #     self_feat = self_feat.to(device)
#             #     x3d = x3d.to(device)
#             #     target = target.to(device)

#             # ----- 핵심: return_attn=True로 호출해서 (pred, attn_dict) 받기
#             pred, attn = model(bg, self_feat, x3d, return_attn=True)

#             loss = criterion(pred, target)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.detach().item()

#             # ----- Attention 확인 (batch 기준 출력 옵션)
#             if log_attn_every_batch is not None and (batch_idx % log_attn_every_batch == 0):
#                 _print_attention_stats(attn, epoch=epoch, batch_idx=batch_idx)

#         train_loss /= len(train_data_loader.dataset)
#         print(f"Epoch {epoch+1}, train loss {train_loss:.4f}")

#         if epoch == max_epochs - 1:
#             _print_attention_stats(attn, epoch=epoch, batch_idx="last")


# @torch.no_grad()
# def _print_attention_stats(attn: dict, epoch, batch_idx):
#     """
#     attn dict에는 {"attn_2d": (B,p2d), "attn_3d": (B,p3d), "hg":..., "hg1":..., "hg2":...}가 있다고 가정.
#     여기서는 attention이 확률분포인지(합=1), top-k가 무엇인지 등을 간단히 확인.
#     """
#     attn2d = attn.get("attn_2d", None)
#     attn3d = attn.get("attn_3d", None)

#     print(f"\n[Attn Check] epoch={epoch+1}, batch={batch_idx}")

#     if attn2d is not None:
#         # (B, p2d)
#         s = attn2d.sum(dim=-1)  # 각 샘플의 확률합
#         print(f"  attn_2d shape={tuple(attn2d.shape)}  sum(mean)={s.mean().item():.4f}  sum(min/max)=({s.min().item():.4f},{s.max().item():.4f})")

#         # top-5 descriptor index
#         topk_val, topk_idx = torch.topk(attn2d, k=min(5, attn2d.size(-1)), dim=-1)
#         print(f"  attn_2d top5 idx (sample0)={topk_idx[0].tolist()}  val={topk_val[0].tolist()}")

#     if attn3d is not None:
#         s = attn3d.sum(dim=-1)
#         print(f"  attn_3d shape={tuple(attn3d.shape)}  sum(mean)={s.mean().item():.4f}  sum(min/max)=({s.min().item():.4f},{s.max().item():.4f})")

#         topk_val, topk_idx = torch.topk(attn3d, k=min(5, attn3d.size(-1)), dim=-1)
#         print(f"  attn_3d top5 idx (sample0)={topk_idx[0].tolist()}  val={topk_val[0].tolist()}\n")


#####
def test_gcn(model, criterion, test_data_loader, accs=None):
    preds = None
    model.eval()

    with torch.no_grad():
        test_loss = 0

        for bg, target in test_data_loader:
            pred = model(bg)
            loss = criterion(pred, target)
            test_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)

        test_loss /= len(test_data_loader.dataset)

        print('Test loss: ' + str(test_loss))

    return test_loss, preds


def test_model(model, criterion, test_data_loader, accs=None):
    preds = None
    model.eval()

    targets = None
    self_feats = None

    with torch.no_grad():
        test_loss = 0

        for bg, self_feat, x3d, target in test_data_loader:
            pred = model(bg, self_feat, x3d)
            # pred, attn = model(bg, self_feat, x3d)

            loss = criterion(pred, target)
            test_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
                targets = target.clone().detach()
                self_feats = self_feat.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)
                self_feats = torch.cat((self_feats, self_feat), dim=0)

        test_loss /= len(test_data_loader.dataset)
        print('Test loss: ' + str(test_loss))

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    self_feats = self_feats.cpu().numpy()
    np.savetxt(r'.\results\result.csv', np.concatenate((targets, preds, self_feats), axis=1), delimiter=',')

    return test_loss, preds

def cross_validation(dataset, model, criterion, num_folds, batch_size, max_epochs, train, test, collate, accs=None):
    num_data_points = len(dataset)
    size_fold = int(len(dataset) / float(num_folds))
    folds = []
    models = []
    optimizers = []
    test_losses = []

    for k in range(0, num_folds - 1):
        folds.append(dataset[k * size_fold:(k + 1) * size_fold])

    folds.append(dataset[(num_folds - 1) * size_fold:num_data_points])

    for k in range(0, num_folds):
        models.append(copy.deepcopy(model))
        optimizers.append(optim.Adam(models[k].parameters(), weight_decay=0.01))

    for k in range(0, num_folds):
        print('--------------- fold {} ---------------'.format(k + 1))

        train_dataset = []
        test_dataset = folds[k]

        for i in range(0, num_folds):
            if i != k:
                train_dataset += folds[i]

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        train(models[k], criterion, optimizers[k], train_data_loader, max_epochs)
        test_loss, pred = test(models[k], criterion, test_data_loader, accs)

        test_losses.append(test_loss)

    df_row = pd.DataFrame([{
        'dataset': DATASET_NAME,
        'epochs': MAX_EPOCHS,
        'num-folds': K,
        'SEED': SEED,
        'criterion': criterion,
        'test_losses': test_losses,
    }])
    csv_path = f'./results./se/{DATASET_NAME}_{MAX_EPOCHS}_{K}_{SEED}_{criterion}.csv'

    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)  # 첫 줄: header 포함
    else:
        df_row.to_csv(csv_path, mode='a', header=False, index=False)  # 이후: append만
   
    if accs is None:
        return np.mean(test_losses)
    else:
        return np.mean(test_losses), np.mean(accs)
