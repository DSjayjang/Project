import torch
import copy
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# for GCN, GAT
def train_model_gcn(model, criterion, optimizer, train_data_loader, max_epochs):   
    train_losses = []
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
        train_losses.append(train_loss)

        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))

    return train_losses


# for EGCN, CONCAT_DS, KROVEX
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


# for GCN, GAT
def collect_train_preds_gcn(model, criterion, train_data_loader):
    preds = None
    model.eval()

    targets = None

    with torch.no_grad():
        train_loss = 0

        for bg, target in train_data_loader:
            pred = model(bg)
            loss = criterion(pred, target)
            train_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
                targets = target.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)

        train_loss /= len(train_data_loader.dataset)

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    np.savetxt(r'.\results\result_train.csv', np.concatenate((targets, preds), axis=1), delimiter=',')


# for EGCN, CONCAT_DS, KROVEX
def collect_train_preds(model, criterion, train_data_loader):
    preds = None
    model.eval()

    targets = None
    self_feats = None

    with torch.no_grad():
        train_loss = 0

        for bg, self_feat, target in train_data_loader:
            pred = model(bg, self_feat)
            loss = criterion(pred, target)
            train_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
                targets = target.clone().detach()
                self_feats = self_feat.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)
                self_feats = torch.cat((self_feats, self_feat), dim=0)

        train_loss /= len(train_data_loader.dataset)

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    self_feats = self_feats.cpu().numpy()
    np.savetxt(r'.\results\result_train.csv', np.concatenate((targets, preds, self_feats), axis=1), delimiter=',')


# for GCN, GAT
def final_train_model_gcn(model, criterion, optimizer, train_data_loader, max_epochs):
    preds = None

    train_losses = []
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

            if preds is None:
                preds = pred.clone().detach()
                targets = target.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)

        train_loss /= len(train_data_loader.dataset)
        train_losses.append(train_loss)

        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))

    preds = preds.detach().cpu().numpy()
    targets = targets.cpu().numpy()
    np.savetxt(r'.\results\result_train.csv', np.concatenate((targets, preds), axis=1), delimiter=',')

    return train_losses


# for EGCN, CONCAT_DS, KROVEX
def final_train_model(model, criterion, optimizer, train_data_loader, max_epochs):
    preds = None

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

            if preds is None:
                preds = pred.clone().detach()
                targets = target.clone().detach()
                self_feats = self_feat.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)
                self_feats = torch.cat((self_feats, self_feat), dim=0)

        train_loss /= len(train_data_loader.dataset)
        train_losses.append(train_loss)

        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))

    preds = preds.detach().cpu().numpy()
    targets = targets.cpu().numpy()
    self_feats = self_feats.cpu().numpy()
    np.savetxt(r'.\results\result_train.csv', np.concatenate((targets, preds, self_feats), axis=1), delimiter=',')

    return train_losses


# for GCN, GAT
def val_model_gcn(model, criterion, val_data_loader, k):
    preds = None
    model.eval()

    targets = None

    with torch.no_grad():
        val_loss = 0

        for bg, target in val_data_loader:
            pred = model(bg)
            loss = criterion(pred, target)
            val_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
                targets = target.clone().detach()
                # self_feats = self_feat.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)

        val_loss /= len(val_data_loader.dataset)

        print('Val loss: ' + str(val_loss))

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    return val_loss, preds


# for EGCN, CONCAT_DS, KROVEX
def val_model(model, criterion, val_data_loader, k):
    preds = None
    model.eval()

    targets = None
    self_feats = None

    with torch.no_grad():
        val_loss = 0

        for bg, self_feat, target in val_data_loader:
            pred = model(bg, self_feat)
            loss = criterion(pred, target)
            val_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
                targets = target.clone().detach()
                self_feats = self_feat.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)
                self_feats = torch.cat((self_feats, self_feat), dim=0)

        val_loss /= len(val_data_loader.dataset)

        print('Val loss: ' + str(val_loss))

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    self_feats = self_feats.cpu().numpy()

    return val_loss, preds


def cross_validation(dataset, model, criterion, num_folds, batch_size, max_epochs, train, val, collate):
    num_data_points = len(dataset)
    size_fold = int(len(dataset) / float(num_folds))
    folds = []
    models = []
    optimizers = []
    val_losses = []

    # best model
    best_model = None
    best_loss = float('inf')

    for k in range(0, num_folds - 1):
        folds.append(dataset[k * size_fold:(k + 1) * size_fold])

    folds.append(dataset[(num_folds - 1) * size_fold:num_data_points])

    for k in range(0, num_folds):
        models.append(copy.deepcopy(model))
        optimizers.append(optim.Adam(models[k].parameters(), weight_decay=0.01))

    fold_train_losses = []
    fold_valid_losses = []

    
    for k in range(num_folds):
        print('--------------- fold {} ---------------'.format(k + 1))

        train_dataset = []
        val_dataset = folds[k]

        for i in range(0, num_folds):
            if i != k:
                train_dataset += folds[i]


        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        # Train the model for this fold
        train_losses = train(models[k], criterion, optimizers[k], train_data_loader, max_epochs)
        fold_train_losses.append(train_losses)

        # Validate the model for this fold
        val_loss, pred = val(models[k], criterion, val_data_loader, k)
        val_losses.append(val_loss)

        # best model 저장 (validation loss 기준으로)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(models[k])
            best_k = k
    print(f'Best Validation Loss: {best_loss}')

    # # Plot fold별 loss
    # fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5), sharey=True)
    # for k in range(num_folds):
    #     epochs = list(range(1, max_epochs + 1))
    #     axes[k].plot(epochs, fold_train_losses[k], label="Train Loss")

    #     axes[k].set_xlabel("Epoch")
    #     axes[k].set_ylabel("Loss")
    #     axes[k].legend()
    #     axes[k].set_title(f"Fold {k+1}")

    # plt.suptitle("Train Loss Across Folds")
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()

    Loss_df = pd.DataFrame({'Fold': list(range(1, len(fold_train_losses) + 1)), 'Train Loss': fold_train_losses})
    Loss_df.to_csv(r'.\results\loss.csv', index = False)

    return np.mean(val_losses), best_model, best_k


# for GCN, GAT
def test_model_gcn(model, criterion, test_data_loader):
    preds = None
    model.eval()

    targets = None

    with torch.no_grad():
        test_loss = 0

        for bg, target in test_data_loader:
            pred = model(bg)
            loss = criterion(pred, target)
            test_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
                targets = target.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)

        test_loss /= len(test_data_loader.dataset)

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    np.savetxt(r'.\results\result_test.csv', np.concatenate((targets, preds), axis=1), delimiter=',')

    return test_loss, preds


# for EGCN, CONCAT_DS, KROVEX
def test_model(model, criterion, test_data_loader):
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

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    self_feats = self_feats.cpu().numpy()
    np.savetxt(r'.\results\result_test.csv', np.concatenate((targets, preds, self_feats), axis=1), delimiter=',')

    return test_loss, preds