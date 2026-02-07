import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 0) Small utilities
# -----------------------------
def safe_softplus(x, beta=1.0, threshold=20.0, eps=1e-8):
    # softplus + epsilon to ensure positivity
    return F.softplus(x, beta=beta, threshold=threshold) + eps

def dirichlet_kl(alpha: torch.Tensor, alpha0: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(alpha0) ) per sample.
    alpha:  (B, D) positive
    alpha0: (1, D) or (B, D) positive
    """
    # log B(alpha) = sum lgamma(alpha) - lgamma(sum alpha)
    sum_alpha = alpha.sum(dim=-1, keepdim=True).clamp_min(eps)
    sum_alpha0 = alpha0.sum(dim=-1, keepdim=True).clamp_min(eps)

    logB_alpha  = torch.lgamma(alpha).sum(dim=-1, keepdim=True) - torch.lgamma(sum_alpha)
    logB_alpha0 = torch.lgamma(alpha0).sum(dim=-1, keepdim=True) - torch.lgamma(sum_alpha0)

    # KL formula
    # KL = log B(alpha0) - log B(alpha) + sum (alpha - alpha0) * (psi(alpha) - psi(sum alpha))
    dig_alpha = torch.digamma(alpha.clamp_min(eps))
    dig_sum   = torch.digamma(sum_alpha)

    t1 = logB_alpha0 - logB_alpha
    t2 = ((alpha - alpha0) * (dig_alpha - dig_sum)).sum(dim=-1, keepdim=True)
    kl = (t1 + t2).squeeze(-1)  # (B,)
    return kl

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Simple scatter mean using index_add.
    src:   (N, d)
    index: (N,) values in [0, dim_size-1]
    """
    out = src.new_zeros((dim_size, src.size(-1)))
    cnt = src.new_zeros((dim_size, 1))
    out.index_add_(0, index, src)
    ones = src.new_ones((src.size(0), 1))
    cnt.index_add_(0, index, ones)
    return out / cnt.clamp_min(1.0)

def graph_batch_mean(bg, node_feat: torch.Tensor) -> torch.Tensor:
    """
    DGLGraph batched readout: mean over nodes per graph.
    bg: batched DGLGraph
    node_feat: (total_nodes, d)
    returns: (B, d)
    """
    # bg.batch_num_nodes() exists in DGL batched graph
    num_nodes = bg.batch_num_nodes()
    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.tolist()

    chunks = torch.split(node_feat, num_nodes, dim=0)
    hg = torch.stack([c.mean(dim=0) if c.numel() > 0 else node_feat.new_zeros(node_feat.size(-1)) for c in chunks], dim=0)
    return hg


# -----------------------------
# 1) Your GCNConv (keep as-is style)
# -----------------------------
class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out)) if bias else None
        self.eps = 1e-12
        nn.init.xavier_uniform_(self.weight)

    def forward(self, g, feat: torch.Tensor):
        # NOTE: g.adj().to_dense() is expensive for big graphs; but you said backbone 유지.
        A_tilde = g.adj().to_dense()
        deg = A_tilde.sum(dim=1)
        D_tilde = torch.pow(deg.clamp_min(self.eps), -0.5)

        X1 = feat * D_tilde.unsqueeze(-1)
        X2 = A_tilde @ X1
        X3 = X2 * D_tilde.unsqueeze(-1)

        out = X3 @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


# -----------------------------
# 2) Graph Encoder (GCN backbone)
# -----------------------------
class GraphEncoderGCN(nn.Module):
    def __init__(self, dim_in: int, d_g: int = 64, hidden: int = 128, drop: float = 0.1):
        super().__init__()
        self.gc1 = GCNConv(dim_in, hidden, bias=True)
        self.gc2 = GCNConv(hidden, d_g, bias=True)
        self.drop = nn.Dropout(drop)
        self.ln = nn.LayerNorm(d_g)

    def forward(self, bg, node_feat: torch.Tensor):
        h = F.relu(self.gc1(bg, node_feat))
        h = self.drop(h)
        h = self.gc2(bg, h)
        h = self.ln(h)
        hg = graph_batch_mean(bg, h)  # (B, d_g)
        return h, hg  # node_emb, graph_emb


# -----------------------------
# 3) 2D Descriptor Tokenizer (B, D) -> (B, D, d_t)
# -----------------------------
class DescriptorTokenizer(nn.Module):
    def __init__(self, d_t: int):
        super().__init__()
        self.proj = nn.Linear(1, d_t)
        self.ln = nn.LayerNorm(d_t)

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        # x2d: (B, D)
        tok = self.proj(x2d.unsqueeze(-1))  # (B, D, d_t)
        tok = self.ln(tok)
        return tok


# -----------------------------
# 4) Evidential Dirichlet Attention over descriptor tokens
# -----------------------------
class EvidentialDirichletAttn(nn.Module):
    """
    Produce Dirichlet concentration alpha (B, D) and attention mean a = alpha/sum(alpha).
    Also provides a KL regularizer to a symmetric Dirichlet prior.
    """
    def __init__(self, d_ctx: int, d_desc: int, hidden: int = 128, prior_strength: float = 0.5):
        super().__init__()
        self.d_desc = d_desc
        self.prior_strength = prior_strength

        # alpha = softplus( MLP([ctx; x2d]) ) + eps
        self.mlp = nn.Sequential(
            nn.Linear(d_ctx + d_desc, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_desc),
        )

    def forward(self, ctx: torch.Tensor, x2d: torch.Tensor):
        """
        ctx: (B, d_ctx)
        x2d: (B, D)
        """
        B, D = x2d.shape
        assert D == self.d_desc, f"d_desc mismatch: got {D}, expected {self.d_desc}"

        logits = self.mlp(torch.cat([ctx, x2d], dim=-1))  # (B, D)
        alpha = safe_softplus(logits)                    # (B, D), positive
        alpha_sum = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attn_mean = alpha / alpha_sum                    # (B, D)

        # evidence: alpha_sum (bigger => more confident about importance allocation)
        evidence = alpha_sum.squeeze(-1)                 # (B,)

        # Dirichlet prior KL regularizer (per sample)
        alpha0 = x2d.new_full((1, D), float(self.prior_strength)).clamp_min(1e-8)
        kl = dirichlet_kl(alpha, alpha0)                 # (B,)

        # entropy of mean attention (diagnostic, not used unless you want)
        ent = -(attn_mean * (attn_mean.clamp_min(1e-12).log())).sum(dim=-1)  # (B,)

        return attn_mean, alpha, evidence, kl, ent


# -----------------------------
# 5) Optional: Substructure (fragment) tokens from node embeddings
# -----------------------------
class SubstructurePool(nn.Module):
    """
    Optional module. If you provide:
      - frag_index: (total_nodes,) global fragment id for each node in the batched graph
      - num_frags: int total fragments in the batch (global indexing)
      - frag_batch_index: (num_frags,) which graph each fragment belongs to (0..B-1)
    then we create:
      - frag_tokens: (num_frags, d_g)
      - frag_summary per graph: (B, d_g) mean over its fragments
    """
    def __init__(self):
        super().__init__()

    def forward(self, node_emb: torch.Tensor, frag_index: torch.Tensor, num_frags: int, frag_batch_index: torch.Tensor, B: int):
        # node_emb: (N, d)
        frag_tokens = scatter_mean(node_emb, frag_index, dim_size=num_frags)  # (num_frags, d)
        frag_summary = scatter_mean(frag_tokens, frag_batch_index, dim_size=B)  # (B, d)
        return frag_tokens, frag_summary


# -----------------------------
# 6) Graph + 2D fusion head (simple, strong baseline)
# -----------------------------
class FusionRegressor(nn.Module):
    def __init__(self, d_g: int, d_t: int, mlp_hidden1: int = 128, mlp_hidden2: int = 32, drop: float = 0.3):
        super().__init__()
        # fused z = [hg; u2d; hg*u2d]  (elementwise interaction)
        self.fc1 = nn.Linear(d_g + d_t + d_t, mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3 = nn.Linear(mlp_hidden2, 1)
        self.bn1 = nn.BatchNorm1d(mlp_hidden1)
        self.bn2 = nn.BatchNorm1d(mlp_hidden2)
        self.drop = nn.Dropout(drop)

        # project hg to d_t for interaction
        self.proj_g = nn.Linear(d_g, d_t)

    def forward(self, hg: torch.Tensor, u2d: torch.Tensor):
        # hg: (B, d_g), u2d: (B, d_t)
        g = self.proj_g(hg)                 # (B, d_t)
        inter = g * u2d                     # (B, d_t)
        z = torch.cat([hg, u2d, inter], dim=-1)

        x = self.drop(F.relu(self.bn1(self.fc1(z))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        y = self.fc3(x)
        return y


# -----------------------------
# 7) Full model: BCEA-GCN (2D only)
# -----------------------------
class Net_EvidentialAttn_2D(nn.Module):
    """
    Forward:
      pred, aux = model(bg, x2d, return_aux=True, node_feat_key='h',
                        frag_index=..., num_frags=..., frag_batch_index=...)
    - bg.ndata[node_feat_key] is node feature (total_nodes, dim_in)
    - x2d: (B, D2)
    Optional substructure:
      frag_index: (total_nodes,) global fragment id per node in batch
      num_frags: int
      frag_batch_index: (num_frags,) graph id for each fragment (0..B-1)
    """
    def __init__(
        self,
        dim_in: int,
        dim_2d_desc: int,
        dim_3d_desc: int,
        d_g: int = 64,
        d_t: int = 32,
        gcn_hidden: int = 128,
        attn_hidden: int = 128,
        prior_strength: float = 0.5,   # <1 tends to encourage sparsity, >1 tends to uniform
        drop: float = 0.3,
        use_substructure: bool = False,
    ):
        super().__init__()
        self.dim_2d_desc = dim_2d_desc
        self.use_substructure = use_substructure

        self.genc = GraphEncoderGCN(dim_in=dim_in, d_g=d_g, hidden=gcn_hidden, drop=drop*0.5)
        self.tokenizer = DescriptorTokenizer(d_t=d_t)

        # context maker: if substructure used, ctx = [hg; h_frag_summary]
        d_ctx = d_g * (2 if use_substructure else 1)
        self.ctx_proj = nn.Sequential(
            nn.Linear(d_ctx, d_g),
            nn.ReLU(),
            nn.LayerNorm(d_g),
        )

        self.evid_attn = EvidentialDirichletAttn(d_ctx=d_g, d_desc=dim_2d_desc, hidden=attn_hidden, prior_strength=prior_strength)
        self.subpool = SubstructurePool()

        self.reg = FusionRegressor(d_g=d_g, d_t=d_t, drop=drop)

    def forward(
        self,
        bg,
        x2d: torch.Tensor,
        x3d: torch.Tensor,
        return_aux: bool = False,
        node_feat_key: str = "feat",
        frag_index: torch.Tensor = None,
        num_frags: int = None,
        frag_batch_index: torch.Tensor = None,
    ):
        # print(bg.ndata.keys())
        node_feat = bg.ndata[node_feat_key]  # (total_nodes, dim_in)
        
        node_emb, hg = self.genc(bg, node_feat)  # (N,d_g), (B,d_g)
        B = hg.size(0)

        # optional substructure summary
        if self.use_substructure and (frag_index is not None) and (num_frags is not None) and (frag_batch_index is not None):
            _, hfrag = self.subpool(node_emb, frag_index, num_frags, frag_batch_index, B=B)  # (B, d_g)
            ctx_raw = torch.cat([hg, hfrag], dim=-1)  # (B, 2*d_g)
        else:
            ctx_raw = hg  # (B, d_g)

        ctx = self.ctx_proj(ctx_raw)  # (B, d_g)

        # 2D tokenization
        tok2d = self.tokenizer(x2d)  # (B, D, d_t)

        # evidential attention (Dirichlet)
        attn_mean, alpha, evidence, kl, ent = self.evid_attn(ctx, x2d)  # (B,D), (B,D), (B,), (B,), (B,)

        # weighted pooling of tokens
        u2d = (attn_mean.unsqueeze(-1) * tok2d).sum(dim=1)  # (B, d_t)

        # prediction
        pred = self.reg(hg, u2d)  # (B,1)

        if not return_aux:
            return pred

        aux = {
            "attn_2d": attn_mean,      # (B, D)
            "alpha_2d": alpha,         # (B, D)
            "evidence_2d": evidence,   # (B,)
            "entropy_2d": ent,         # (B,)
            # 이걸 네 루프의 lambda_desc * aux["nll_2d"].mean()에 바로 사용
            # (정당화: Dirichlet posterior를 sparse/uniform prior에 붙이는 KL regularizer)
            "nll_2d": kl,              # (B,)
        }
        return pred, aux
