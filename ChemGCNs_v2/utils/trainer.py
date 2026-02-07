import torch
import copy
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import pandas as pd
import matplotlib.pyplot as plt

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
    train_losses = []
    model.train()

    for epoch in range(0, max_epochs):
        train_loss = 0

        for bg, self_feat, target in train_data_loader:
            pred = model(bg, self_feat)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        train_loss /= len(train_data_loader.dataset)
        train_losses.append(train_loss)
        
        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))

    return train_losses

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

        for bg, self_feat, target in test_data_loader:
            pred = model(bg, self_feat)
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

def cross_validation(dataset, model, criterion, num_folds, batch_size, max_epochs, train, test, collate, model_name, accs=None):
    num_data_points = len(dataset)
    size_fold = int(len(dataset) / float(num_folds))
    folds = []
    models = []
    optimizers = []
    test_losses = []
    # LR = 0.001
    
    for k in range(0, num_folds - 1):
        folds.append(dataset[k * size_fold:(k + 1) * size_fold])

    folds.append(dataset[(num_folds - 1) * size_fold:num_data_points])

    for k in range(0, num_folds):
        models.append(copy.deepcopy(model))
        optimizers.append(optim.Adam(models[k].parameters(), weight_decay=0.01))

    fold_train_losses = []
    fold_valid_losses = []

    for k in range(0, num_folds):
        print('--------------- fold {} ---------------'.format(k + 1))

        train_dataset = []
        test_dataset = folds[k]

        for i in range(0, num_folds):
            if i != k:
                train_dataset += folds[i]

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        train_losses = train(models[k], criterion, optimizers[k], train_data_loader, max_epochs)
        fold_train_losses.append(train_losses)

        test_loss, pred = test(models[k], criterion, test_data_loader, accs)

        test_losses.append(test_loss)

    # Plot fold별 loss
    save_path = f'./results/loss/{DATASET_NAME}_{model_name}_{MAX_EPOCHS}_{SEED}_{criterion}.png'
    fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5), sharey=True)
    for k in range(num_folds):
        epochs = list(range(1, max_epochs + 1))
        axes[k].plot(epochs, fold_train_losses[k], label="Train Loss")

        axes[k].set_xlabel("Epoch")
        axes[k].set_ylabel("Loss")
        axes[k].legend()
        axes[k].set_title(f"Fold {k+1}")

    plt.suptitle("Train Loss Across Folds")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()
    # save
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
