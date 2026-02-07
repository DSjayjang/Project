import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import torch.optim as optim

lambda_desc = 0.001

def train(model, criterion, optimizer, train_loader, max_epochs):
    for epoch in range(max_epochs):
        train_loss_sum = 0.0
        train_n = 0

        model.train()
        for bg, feat_2d, feat_3d, target in train_loader:
            pred, aux = model(bg, feat_2d, feat_3d, return_aux=True)

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

        print(f'[FULL TRAIN] Epoch {epoch+1} | loss {train_loss:.4f}')


def test(model, criterion, test_loader, model_name):
    model.eval()
    loss_sum = 0.0
    test_n = 0
    desc_sum = 0.0    # optional: log only

    preds, targets = [], []

    with torch.no_grad():
        for bg, feat_2d, feat_3d, target in test_loader:
            pred, aux = model(bg, feat_2d, feat_3d, return_aux = True)

            y_loss = criterion(pred, target)

            bs = target.size(0)
            loss_sum +=y_loss.detach().item() * bs
            test_n += bs

            # aux loss
            if aux is not None and 'nll_2d' in aux:
                dloss = aux['nll_2d'].mean().item()
                # print('nll_2d', aux['nll_2d'])
            else:
                dloss = 0.0
                # print('not nll_2d', aux['nll_2d'])
            desc_sum += dloss * bs

            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())

    test_loss = loss_sum / test_n

    y_true = torch.cat(targets, dim=0).numpy().squeeze()
    y_pred = torch.cat(preds, dim=0).numpy().squeeze()
    test_r2 = float(r2_score(y_true, y_pred))

    print(f'[FINAL TEST] loss {test_loss:.4f} | R2 {test_r2:.2f}')

    targets_np = torch.cat(targets, dim=0).cpu().numpy()
    preds_np = torch.cat(preds, dim=0).cpu().numpy()

    np.savetxt(rf'.\results\result_test_{model_name}_{criterion}.csv', np.concatenate((targets_np, preds_np), axis=1), 
               delimiter=',', header='target,pred',comments='')

    return test_loss, test_r2


def collect_train_preds(model, train_dataset, batch_size, collate_fn, model_name, criterion):
    model.eval()
    preds, targets = [], []

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for bg, feat_2d, feat_3d, target in train_eval_loader:
            pred = model(bg, feat_2d, feat_3d)

            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())

    targets_np = torch.cat(targets, dim=0).numpy()
    preds_np = torch.cat(preds, dim=0).numpy()

    np.savetxt(rf'.\results\result_train_{model_name}_{criterion}.csv', np.concatenate((targets_np, preds_np), axis=1),
        delimiter=',', header='target,pred',comments='')


def full_train_and_test(train_dataset, test_dataset, model, criterion, batch_size, max_epochs, collate_fn, model_name, lr=1e-3, weight_decay=0.01):
    m = copy.deepcopy(model)
    opt = optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)

    # train
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    train(m, criterion, opt, train_loader, max_epochs)
    collect_train_preds(m, train_dataset, batch_size, collate_fn, model_name, criterion)

    # test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    test_loss, test_r2 = test(m, criterion, test_loader, model_name)

    return test_loss, test_r2
