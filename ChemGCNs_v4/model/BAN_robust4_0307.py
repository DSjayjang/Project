import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


# =========================================================
# Gradient Reversal Layer
# =========================================================
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x, lambda_grl=1.0):
    return GRL.apply(x, lambda_grl)


# =========================================================
# GCN layer
# =========================================================
class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out)) if bias else None
        self.eps = 1e-12
        nn.init.xavier_uniform_(self.weight)

    def forward(self, g, feat: torch.Tensor):
        A_tilde = g.adj().to_dense().to(feat.device)
        deg = A_tilde.sum(dim=1)
        D_tilde = torch.pow(deg.clamp_min(self.eps), -0.5)

        X1 = feat * D_tilde.unsqueeze(-1)
        X2 = A_tilde @ X1
        X3 = X2 * D_tilde.unsqueeze(-1)

        out = X3 @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


# =========================================================
# Descriptor tokenizer
# =========================================================
class DescriptorTokenizer(nn.Module):
    def __init__(self, d_desc: int, d_t: int):
        super().__init__()
        self.id_embedding = nn.Embedding(d_desc, d_t)
        self.val_embedding = nn.Sequential(
            nn.Linear(1, d_t // 2),
            nn.ReLU(),
            nn.Linear(d_t // 2, d_t)
        )
        self.norm = nn.LayerNorm(d_t)

    def forward(self, descriptor: torch.Tensor):
        bs, d_desc = descriptor.shape

        desc = descriptor.unsqueeze(-1)                   # (bs, d_desc, 1)
        val_emb = self.val_embedding(desc)                # (bs, d_desc, d_t)

        ids = torch.arange(d_desc, device=descriptor.device)
        id_emb = self.id_embedding(ids)                   # (d_desc, d_t)
        id_emb = id_emb.unsqueeze(0).expand(bs, -1, -1)  # (bs, d_desc, d_t)

        desc_tks = self.norm(val_emb + id_emb)
        return desc_tks


# =========================================================
# Token masking for robustness
# =========================================================
class DescriptorTokenDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.training) or self.p <= 0:
            return x
        keep_mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > self.p).float()
        return x * keep_mask


# =========================================================
# Dual-path encoder
# =========================================================
class DualPathEncoder(nn.Module):
    def __init__(self, dim_graph: int, hidden_ratio: float = 1.0, negative_slope: float = 0.1):
        super().__init__()
        hidden_dim = int(dim_graph * hidden_ratio)

        self.spec_proj = nn.Sequential(
            nn.Linear(dim_graph, hidden_dim, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, dim_graph, bias=False),
        )
        self.inv_proj = nn.Sequential(
            nn.Linear(dim_graph, hidden_dim, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, dim_graph, bias=False),
        )

        self.norm_spec = nn.LayerNorm(dim_graph)
        self.norm_inv = nn.LayerNorm(dim_graph)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

        self.gate = nn.Sequential(
            nn.Linear(dim_graph, dim_graph),
            nn.ReLU(),
            nn.Linear(dim_graph, dim_graph)
        )

    def forward(self, hg: torch.Tensor):
        h_spec = self.norm_spec(hg + self.spec_proj(hg))
        h_inv = self.norm_inv(hg + self.inv_proj(hg))
        # gate = torch.sigmoid(self.gate(hg))   # (bs, dim_graph)
        # h_spec = self.norm_spec(gate * hg)
        # h_inv  = self.norm_inv((1.0 - gate) * hg)
        return h_spec, h_inv

    @staticmethod
    def orth_loss(h_spec: torch.Tensor, h_inv: torch.Tensor):
        h_s = F.normalize(h_spec, dim=-1)
        h_i = F.normalize(h_inv, dim=-1)
        cos_sim = (h_s * h_i).sum(dim=-1)
        return cos_sim.pow(2).mean()
        # h_spec = h_spec - h_spec.mean(dim=0, keepdim=True)
        # h_inv = h_inv - h_inv.mean(dim=0, keepdim=True)
        # C = (h_spec.T @ h_inv) / h_spec.size(0)
        # return (C ** 2).mean()

# =========================================================
# Scaffold-specific classifier
# =========================================================
class ScaffoldSpecificHead(nn.Module):
    def __init__(self, dim_graph: int, n_scaffolds: int):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(dim_graph, dim_graph),
            nn.ReLU(),
            nn.Linear(dim_graph, n_scaffolds)
        )

    def forward(self, h_spec: torch.Tensor):
        return self.clf(h_spec)

    def loss(self, h_spec: torch.Tensor, scaffold_ids: torch.Tensor):
        logits = self.forward(h_spec)
        return F.cross_entropy(logits, scaffold_ids)


# =========================================================
# Scaffold adversary on invariant branch
# =========================================================
class ScaffoldAdversary(nn.Module):
    def __init__(self, dim_graph: int, n_scaffolds: int):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(dim_graph, dim_graph),
            nn.ReLU(),
            nn.Linear(dim_graph, n_scaffolds)
        )

    def forward(self, h_inv: torch.Tensor, lambda_grl: float = 1.0):
        h_rev = grad_reverse(h_inv, lambda_grl)
        return self.clf(h_rev)

    def loss(self, h_inv: torch.Tensor, scaffold_ids: torch.Tensor, lambda_grl: float = 1.0):
        logits = self.forward(h_inv, lambda_grl=lambda_grl)
        return F.cross_entropy(logits, scaffold_ids)


# =========================================================
# Bilinear attention
# =========================================================
class BilinearAttention(nn.Module):
    def __init__(self, d_q: int, d_t: int, glimpse: int, K: int = None, bias=False):
        super().__init__()

        self.d_q = d_q
        self.d_t = d_t
        self.glimpse = glimpse

        self.K_joint = d_t if K is None else K
        self.K_attn = self.K_joint

        self.U_attn = nn.Linear(d_q, self.K_attn, bias=bias)
        self.V_attn = nn.Linear(d_t, self.K_attn, bias=bias)
        self.p = nn.Parameter(torch.empty(self.glimpse, self.K_attn))
        nn.init.normal_(self.p, mean=0, std=1.0 / math.sqrt(self.K_attn))

        self.U_joint = nn.Linear(d_q, self.K_joint, bias=bias)
        self.V_joint = nn.Linear(d_t, self.K_joint, bias=bias)
        self.P = nn.Linear(self.K_joint, d_q, bias=bias)

        self.res_norm = nn.ModuleList([nn.LayerNorm(d_q) for _ in range(self.glimpse)])

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        bs, M, d_t = Y.shape
        f_i = X

        YV_attn = F.relu(self.V_attn(Y))
        YV_joint = F.relu(self.V_joint(Y))

        attn_list = []

        for g in range(self.glimpse):
            XU_attn = F.relu(self.U_attn(f_i))
            logits = torch.einsum('bk,bmk,k->bm', XU_attn, YV_attn, self.p[g])

            attn = F.softmax(logits, dim=-1)
            attn_list.append(attn)

            Vy_hat = torch.einsum('bm,bmk->bk', attn, YV_joint)
            XU_joint = F.relu(self.U_joint(f_i))
            f_joint = self.P(XU_joint * Vy_hat)

            f_i = self.res_norm[g](f_i + f_joint)

        return f_i, attn_list


# =========================================================
# Main network
# =========================================================
class Net(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_desc_2d: int,
        n_scaffolds: int = None,
        grl_lambda: float = 1.0,
    ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_desc_2d = dim_desc_2d
        self.use_scaffold = n_scaffolds is not None
        self.grl_lambda = grl_lambda

        # hyperparameters
        self.dim_graph = 64 # 64
        self.dim_fc1 = 256 # 256
        self.dim_fc2 = 64 # 64
        self.dim_out = 1
        self.drop_out = 0.2
        self.d_t = 64 # 64
        self.K = 32 # 32
        self.glimpse = 2

        # graph encoder
        self.gc1 = GCNConv(dim_in, 128)
        self.gc2 = GCNConv(128, self.dim_graph)

        # tokenizer
        self.tokenizer = DescriptorTokenizer(self.dim_desc_2d, self.d_t)
        self.token_dropout = DescriptorTokenDropout(p=0.1)

        # dual path
        self.dual_encoder = DualPathEncoder(self.dim_graph)

        # fusion
        self.ban = BilinearAttention(
            d_q=self.dim_graph,
            d_t=self.d_t,
            glimpse=self.glimpse,
            K=self.K
        )

        # heads
        self.fc1 = nn.Linear(self.dim_graph, self.dim_fc1)
        self.fc2 = nn.Linear(self.dim_fc1, self.dim_fc2)
        self.fc3 = nn.Linear(self.dim_fc2, self.dim_out)

        self.bn1 = nn.LayerNorm(self.dim_fc1)
        self.bn2 = nn.LayerNorm(self.dim_fc2)
        self.dropout = nn.Dropout(self.drop_out)

        if self.use_scaffold:
            self.scaffold_specific = ScaffoldSpecificHead(self.dim_graph, n_scaffolds)
            self.scaffold_adversary = ScaffoldAdversary(self.dim_graph, n_scaffolds)
        else:
            self.scaffold_specific = None
            self.scaffold_adversary = None

    def encode_graph(self, g):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return hg

    def forward(self, g, desc_2d):
        hg = self.encode_graph(g)

        # descriptor tokens
        desc_tks = self.tokenizer(desc_2d)
        desc_tks = self.token_dropout(desc_tks)

        # dual path
        h_spec, h_inv = self.dual_encoder(hg)

        # invariant branch is used for property prediction
        fused, attn_list = self.ban(h_inv, desc_tks)

        out = F.relu(self.bn1(self.fc1(fused)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)

        return out, attn_list, h_spec, h_inv

    def compute_scaffold_losses(
        self,
        h_spec: torch.Tensor,
        h_inv: torch.Tensor,
        scaffold_ids: torch.Tensor = None,
        lambda_orth: float = 0.1, # 0.1
        lambda_spec: float = 0.5, # 0.5
        lambda_adv: float = 0.2, # 0.5
    ):
        device = h_inv.device
        zero = torch.tensor(0.0, device=device)

        losses = {
            "orth": zero,
            "spec": zero,
            "adv": zero,
            "total": zero,
        }

        if (not self.use_scaffold) or (scaffold_ids is None):
            return losses

        losses["orth"] = DualPathEncoder.orth_loss(h_spec, h_inv)
        losses["spec"] = self.scaffold_specific.loss(h_spec, scaffold_ids)
        losses["adv"] = self.scaffold_adversary.loss(
            h_inv, scaffold_ids, lambda_grl=self.grl_lambda
        )

        losses["total"] = (
            lambda_orth * losses["orth"] +
            lambda_spec * losses["spec"] +
            lambda_adv * losses["adv"]
        )
        return losses