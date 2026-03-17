import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Toy descriptor family specification
# ============================================================
# We mimic the proposed 5-family RDKit grouping, but with small toy dimensions.
# In a real setting, these indices would correspond to the real 196 RDKit descriptors.

FAMILY_SPECS: Dict[str, int] = {
    "constitutional": 6,
    "topological": 5,
    "physicochemical": 4,
    "electronic": 5,
    "fragment": 8,
}

FAMILY_NAMES = list(FAMILY_SPECS.keys())
TOTAL_DESC_DIM = sum(FAMILY_SPECS.values())


def build_family_slices(specs: Dict[str, int]) -> Dict[str, slice]:
    start = 0
    out = {}
    for name, dim in specs.items():
        out[name] = slice(start, start + dim)
        start += dim
    return out


FAMILY_SLICES = build_family_slices(FAMILY_SPECS)


# ============================================================
# 2. Toy molecular graph generator
# ============================================================
# We generate small random graphs and synthetic descriptor vectors.
# The regression target is deliberately constructed to depend on
# graph structure + family-level descriptor statistics so that the
# model can learn meaningful family-wise interactions.
# ============================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ToySample:
    x: torch.Tensor           # [N, node_dim]
    adj: torch.Tensor         # [N, N]
    descriptors: torch.Tensor # [D]
    y: torch.Tensor           # [1]
    n_nodes: int


class ToyMolDataset(Dataset):
    def __init__(self, n_samples: int, node_dim: int = 8, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.samples: List[ToySample] = []
        for _ in range(n_samples):
            n = int(torch.randint(low=4, high=9, size=(1,), generator=g).item())  # 4~8 nodes
            x = torch.randn(n, node_dim, generator=g)

            # random undirected adjacency
            adj = torch.zeros(n, n)
            for i in range(n):
                for j in range(i + 1, n):
                    if torch.rand(1, generator=g).item() < 0.35:
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0
            # ensure connectivity-ish by linking chain if isolated
            for i in range(n - 1):
                adj[i, i + 1] = max(adj[i, i + 1], 1.0)
                adj[i + 1, i] = max(adj[i + 1, i], 1.0)
            adj.fill_diagonal_(1.0)

            descriptors = torch.randn(TOTAL_DESC_DIM, generator=g)

            # handcrafted synthetic target using both graph and family descriptors
            deg = adj.sum(dim=1)
            graph_signal = 0.25 * x[:, 0].sum() + 0.15 * deg.mean() + 0.10 * x[:, 1].mean()

            d_const = descriptors[FAMILY_SLICES["constitutional"]].mean()
            d_topo = descriptors[FAMILY_SLICES["topological"]].sum() / FAMILY_SPECS["topological"]
            d_phys = descriptors[FAMILY_SLICES["physicochemical"]].mean()
            d_elec = descriptors[FAMILY_SLICES["electronic"]].mean()
            d_frag = descriptors[FAMILY_SLICES["fragment"]].mean()

            # interaction-style target
            y = (
                0.6 * graph_signal
                + 0.8 * d_const
                + 0.5 * d_topo * deg.mean()
                + 0.7 * d_phys * x[:, 2].mean()
                + 0.9 * d_elec * x[:, 3].sum() / n
                + 0.6 * d_frag * (deg.max() / n)
            )
            y = y + 0.05 * torch.randn(1, generator=g)

            self.samples.append(ToySample(x=x, adj=adj, descriptors=descriptors, y=y.view(1), n_nodes=n))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ToySample:
        return self.samples[idx]


# ============================================================
# 3. Collate with padding and node mask
# ============================================================


def collate_toy(batch: List[ToySample]) -> Dict[str, torch.Tensor]:
    bs = len(batch)
    max_n = max(s.n_nodes for s in batch)
    node_dim = batch[0].x.size(1)

    x = torch.zeros(bs, max_n, node_dim)
    adj = torch.zeros(bs, max_n, max_n)
    mask = torch.zeros(bs, max_n, dtype=torch.bool)
    descriptors = torch.stack([s.descriptors for s in batch], dim=0)
    y = torch.stack([s.y for s in batch], dim=0)

    for b, s in enumerate(batch):
        n = s.n_nodes
        x[b, :n] = s.x
        adj[b, :n, :n] = s.adj
        mask[b, :n] = True

    return {
        "x": x,
        "adj": adj,
        "mask": mask,
        "descriptors": descriptors,
        "y": y,
    }


# ============================================================
# 4. Graph encoder (manual 2-layer GCN)
# ============================================================


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask_f = mask.float()
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)
    num = (x * mask_f).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp_min(1.0)
    return num / den


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, Din], adj: [B, N, N]
        deg = adj.sum(dim=-1)                            # [B, N]
        deg_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)     # [B, N]
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.bmm(norm_adj, x)
        out = self.lin(out)
        out = out * mask.unsqueeze(-1).float()
        return out


class ToyGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.gcn1 = GCNLayer(node_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, emb_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.gcn1(x, adj, mask))
        h = self.gcn2(h, adj, mask)  # [B, N, d]
        g = masked_mean(h, mask, dim=1)  # [B, d]
        return h, g


# ============================================================
# 5. Family tokenization: z_k = phi_k(d^(k))
# ============================================================


class FamilyTokenizer(nn.Module):
    def __init__(self, family_specs: Dict[str, int], emb_dim: int):
        super().__init__()
        self.family_names = list(family_specs.keys())
        self.family_slices = build_family_slices(family_specs)
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(family_specs[name], emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            for name in self.family_names
        })

    def forward(self, descriptors: torch.Tensor) -> Dict[str, torch.Tensor]:
        # descriptors: [B, D]
        z = {}
        for name in self.family_names:
            d_k = descriptors[:, self.family_slices[name]]  # [B, p_k]
            z[name] = self.encoders[name](d_k)              # [B, d]
        return z


# ============================================================
# 6. Family-specific low-rank bilinear node attention
#    e_{k,i} = (U_k z_k)^T (V_k h_i) / sqrt(r)
# ============================================================


class FamilyBilinearAttention(nn.Module):
    def __init__(self, emb_dim: int, rank: int):
        super().__init__()
        self.q_proj = nn.Linear(emb_dim, rank, bias=False)
        self.k_proj = nn.Linear(emb_dim, rank, bias=False)
        self.scale = math.sqrt(rank)

    def forward(self, z_k: torch.Tensor, H: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z_k: [B, d], H: [B, N, d], mask: [B, N]
        q = self.q_proj(z_k)                    # [B, r]
        k = self.k_proj(H)                      # [B, N, r]
        scores = torch.einsum("br,bnr->bn", q, k) / self.scale  # [B, N]
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)  # [B, N]
        c_k = torch.einsum("bn,bnd->bd", alpha, H)              # [B, d]
        return alpha, c_k


# ============================================================
# 7. Family fusion: f_k = psi_k([c_k || z_k || c_k*z_k || c_k-z_k])
# ============================================================


class FamilyFusion(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, c_k: torch.Tensor, z_k: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([c_k, z_k, c_k * z_k, c_k - z_k], dim=-1)
        return self.mlp(fused)


# ============================================================
# 8. Family aggregation: beta_k = softmax(a^T tanh(W_f f_k + W_g g))
# ============================================================


class FamilyAggregator(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.Wf = nn.Linear(emb_dim, emb_dim)
        self.Wg = nn.Linear(emb_dim, emb_dim)
        self.a = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, family_reps: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # family_reps: [B, K, d], g: [B, d]
        g_exp = g.unsqueeze(1).expand_as(family_reps)
        scores = self.a(torch.tanh(self.Wf(family_reps) + self.Wg(g_exp))).squeeze(-1)  # [B, K]
        beta = torch.softmax(scores, dim=-1)
        h_fam = torch.einsum("bk,bkd->bd", beta, family_reps)
        return beta, h_fam


# ============================================================
# 9. Full FaBiG toy model
# ============================================================


class FaBiGToy(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 32, emb_dim: int = 32, rank: int = 8):
        super().__init__()
        self.family_names = FAMILY_NAMES
        self.graph_encoder = ToyGraphEncoder(node_dim=node_dim, hidden_dim=hidden_dim, emb_dim=emb_dim)
        self.tokenizer = FamilyTokenizer(FAMILY_SPECS, emb_dim)
        self.attn = nn.ModuleDict({name: FamilyBilinearAttention(emb_dim, rank) for name in self.family_names})
        self.fusion = nn.ModuleDict({name: FamilyFusion(emb_dim) for name in self.family_names})
        self.aggregator = FamilyAggregator(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor, descriptors: torch.Tensor):
        # 1) Graph encoder
        H, g = self.graph_encoder(x, adj, mask)

        # 2) Family tokenization
        z_dict = self.tokenizer(descriptors)

        # 3) Family-specific bilinear attention + fusion
        family_reps = []
        attn_maps = {}
        contexts = {}
        for name in self.family_names:
            z_k = z_dict[name]
            alpha_k, c_k = self.attn[name](z_k, H, mask)
            f_k = self.fusion[name](c_k, z_k)
            family_reps.append(f_k)
            attn_maps[name] = alpha_k
            contexts[name] = c_k

        family_reps = torch.stack(family_reps, dim=1)  # [B, K, d]

        # 4) Family aggregation
        beta, h_fam = self.aggregator(family_reps, g)

        # 5) Prediction
        h_final = torch.cat([g, h_fam], dim=-1)
        y_hat = self.head(h_final)

        return {
            "y_hat": y_hat,
            "H": H,
            "g": g,
            "z_dict": z_dict,
            "contexts": contexts,
            "attn_maps": attn_maps,
            "family_reps": family_reps,
            "beta": beta,
            "h_fam": h_fam,
            "h_final": h_final,
        }


# ============================================================
# 10. Training / demonstration utilities
# ============================================================


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        x = batch["x"].to(device)
        adj = batch["adj"].to(device)
        mask = batch["mask"].to(device)
        descriptors = batch["descriptors"].to(device)
        y = batch["y"].to(device)

        out = model(x, adj, mask, descriptors)
        loss = F.mse_loss(out["y_hat"], y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        x = batch["x"].to(device)
        adj = batch["adj"].to(device)
        mask = batch["mask"].to(device)
        descriptors = batch["descriptors"].to(device)
        y = batch["y"].to(device)

        out = model(x, adj, mask, descriptors)
        loss = F.mse_loss(out["y_hat"], y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / total_n


@torch.no_grad()
def inspect_single_batch(model, loader, device):
    model.eval()
    batch = next(iter(loader))
    x = batch["x"].to(device)
    adj = batch["adj"].to(device)
    mask = batch["mask"].to(device)
    descriptors = batch["descriptors"].to(device)

    out = model(x, adj, mask, descriptors)

    print("\n==================== SHAPE INSPECTION ====================")
    print(f"x                     : {tuple(x.shape)}")
    print(f"adj                   : {tuple(adj.shape)}")
    print(f"mask                  : {tuple(mask.shape)}")
    print(f"descriptors           : {tuple(descriptors.shape)}")
    print(f"H (node embeddings)   : {tuple(out['H'].shape)}")
    print(f"g (graph embedding)   : {tuple(out['g'].shape)}")

    for name in FAMILY_NAMES:
        print(f"z[{name}]              : {tuple(out['z_dict'][name].shape)}")
        print(f"alpha[{name}]          : {tuple(out['attn_maps'][name].shape)}")
        print(f"c[{name}]              : {tuple(out['contexts'][name].shape)}")

    print(f"family_reps           : {tuple(out['family_reps'].shape)}")
    print(f"beta (family weights) : {tuple(out['beta'].shape)}")
    print(f"h_fam                 : {tuple(out['h_fam'].shape)}")
    print(f"h_final               : {tuple(out['h_final'].shape)}")
    print(f"y_hat                 : {tuple(out['y_hat'].shape)}")

    # show first sample's family weights
    print("\nFirst sample family-level weights beta:")
    for name, w in zip(FAMILY_NAMES, out["beta"][0].cpu().tolist()):
        print(f"  {name:16s}: {w:.4f}")

    # show first sample's node attention per family for valid nodes only
    valid_n = int(mask[0].sum().item())
    print(f"\nFirst sample valid node count: {valid_n}")
    print("First sample node-level attention maps:")
    for name in FAMILY_NAMES:
        alpha = out["attn_maps"][name][0, :valid_n].cpu().tolist()
        alpha_str = ", ".join(f"{a:.3f}" for a in alpha)
        print(f"  {name:16s}: [{alpha_str}]")


# ============================================================
# 11. Main demonstration
# ============================================================


def main():
    set_seed(42)
    device = torch.device("cpu")

    train_ds = ToyMolDataset(n_samples=256, node_dim=8, seed=42)
    val_ds = ToyMolDataset(n_samples=64, node_dim=8, seed=777)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_toy)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_toy)

    model = FaBiGToy(node_dim=8, hidden_dim=32, emb_dim=32, rank=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training FaBiG toy model...")
    for epoch in range(1, 6):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

    inspect_single_batch(model, val_loader, device)


if __name__ == "__main__":
    main()
