import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from model import CrossAttn_TFN
import torch
import torch.nn as nn

criterion = nn.MSELoss()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for g, desc2d, desc3d, y in loader:
        g = g.to(device)
        desc2d = desc2d.to(device)
        desc3d = desc3d.to(device)
        y = y.to(device).float().view(-1, 1)

        pred = model(g, desc2d, desc3d)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    for g, desc2d, desc3d, y in loader:
        g = g.to(device)
        desc2d = desc2d.to(device)
        desc3d = desc3d.to(device)
        y = y.to(device).float().view(-1, 1)

        pred = model(g, desc2d, desc3d)
        loss = criterion(pred, y)

        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)

def grid_search_kfold(dataset, dim_in, dim_2d_desc, dim_3d_desc,
                      param_list, K=5, batch_size=64, epochs=30,
                      collate_fn=None, device="cuda"):
    lr=1e-3
    weight_decay=1e-4

    kf = KFold(n_splits=K, shuffle=True, random_state=0)
    results = []

    for cfg_id, cfg in enumerate(param_list):
        fold_losses = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(len(dataset)))):
            train_ds = Subset(dataset, tr_idx)
            val_ds   = Subset(dataset, va_idx)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn, drop_last=False)
            val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                      collate_fn=collate_fn, drop_last=False)

            model = CrossAttn_TFN.Net_2d_grid(
                dim_in=dim_in,
                dim_2d_desc=dim_2d_desc,
                dim_3d_desc=dim_3d_desc,
                d_t=cfg["d_t"],
                d_k=cfg["d_k"],
                dim_out_fc1=cfg["fc1"],
                dim_out_fc2=cfg["fc2"],
                drop_out=cfg["dropout"],
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            for ep in range(epochs):
                loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                print(f"[debug] fold={fold} ep={ep+1}/{epochs} train_loss={loss:.4f}")

            val_loss = eval_loss(model, val_loader, criterion, device)
            fold_losses.append(val_loss)

        mean_loss = float(np.mean(fold_losses))
        std_loss  = float(np.std(fold_losses))

        results.append({"cfg": cfg, "mean_loss": mean_loss, "std_loss": std_loss})
        print(f"[{cfg_id+1}/{len(param_list)}] cfg={cfg}  val_loss={mean_loss:.4f}±{std_loss:.4f}")

    results.sort(key=lambda x: x["mean_loss"])

    best = results[0]
    print("\n================ BEST (min mean val loss) ================")
    print("Best cfg:", best["cfg"])
    print(f"Mean val loss: {best['mean_loss']:.6f}  Std: {best['std_loss']:.6f}")
    print("==========================================================\n")
    
    return results
