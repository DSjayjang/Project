import torch
import copy
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.mol_conv_scaffold as mc

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

        for bg, self_feat, target in train_data_loader:
            pred = model(bg, self_feat)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        train_loss /= len(train_data_loader.dataset)

        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))


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

def cross_validation(dataset, model, criterion, folds, num_folds, batch_size, max_epochs, train, test, collate, accs=None):
    models = []
    optimizers = []
    test_losses = []

    # folds = mc.scaffold_kfold_split(smiles_list, num_folds)

    for k in range(0, num_folds):
        models.append(copy.deepcopy(model))
        optimizers.append(optim.Adam(models[k].parameters(), weight_decay=0.01))

    for k in range(0, num_folds):
        print('--------------- Scaffold Fold {} ---------------'.format(k + 1))

        test_idx = folds[k]
        train_idx = [idx for i in range(num_folds) if i !=k for idx in folds[i]]

        train_dataset = [dataset[idx] for idx in train_idx]
        test_dataset = [dataset[idx] for idx in test_idx]
        print('len(train_idx)', len(train_idx))
        print('len(teset_idx)', len(test_idx))

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        train(models[k], criterion, optimizers[k], train_data_loader, max_epochs)
        test_loss, pred = test(models[k], criterion, test_data_loader, accs)
        test_losses.append(test_loss)

    if accs is None:
        return np.mean(test_losses)
    else:
        return np.mean(test_losses), np.mean(accs)
