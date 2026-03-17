import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


# =========================================================
# 1. Simple GCNConv
# =========================================================
class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out)) if bias else None
        self.eps = 1e-12
        nn.init.xavier_uniform_(self.weight)

    def forward(self, g, feat: torch.Tensor):
        """
        g    : DGLGraph (batched)
        feat : (N, dim_in)
        """
        A_tilde = g.adj().to_dense().to(feat.device)   # (N, N)
        deg = A_tilde.sum(dim=1)                       # (N,)
        D_tilde = torch.pow(deg.clamp_min(self.eps), -0.5)

        X1 = feat * D_tilde.unsqueeze(-1)
        X2 = A_tilde @ X1
        X3 = X2 * D_tilde.unsqueeze(-1)

        out = X3 @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


# =========================================================
# 2. Descriptor Tokenizer
#    descriptor scalar -> token embedding
# =========================================================
class DescriptorTokenizer(nn.Module):
    def __init__(self, num_desc: int, d_t: int):
        super().__init__()
        self.num_desc = num_desc
        self.d_t = d_t

        # descriptor ID embedding
        self.id_embedding = nn.Embedding(num_desc, d_t)

        # scalar value embedding
        self.value_proj = nn.Sequential(
            nn.Linear(1, d_t),
            nn.GELU(),
            nn.Linear(d_t, d_t)
        )

        self.norm = nn.LayerNorm(d_t)

    def forward(self, feat_desc: torch.Tensor):
        """
        feat_desc: (B, M)  [M = num_desc]
        return   : (B, M, d_t)
        """
        B, M = feat_desc.shape
        assert M == self.num_desc, f"Expected num_desc={self.num_desc}, got {M}"

        device = feat_desc.device
        desc_ids = torch.arange(M, device=device).unsqueeze(0).expand(B, M)  # (B, M)

        id_emb = self.id_embedding(desc_ids)                     # (B, M, d_t)
        val_emb = self.value_proj(feat_desc.unsqueeze(-1))      # (B, M, d_t)

        tokens = self.norm(id_emb + val_emb)
        return tokens


# =========================================================
# 3. Gradient Reversal Layer
# =========================================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)


# =========================================================
# 4. Dual Path Encoder
#    hg -> h_scaf, h_phys
# =========================================================
class DualPathEncoder(nn.Module):
    def __init__(self, dim_graph: int, dropout: float = 0.1):
        super().__init__()
        self.proj_scaf = nn.Linear(dim_graph, dim_graph, bias=False)
        self.proj_phys = nn.Linear(dim_graph, dim_graph, bias=False)

        self.norm_scaf = nn.LayerNorm(dim_graph)
        self.norm_phys = nn.LayerNorm(dim_graph)
        self.dropout = nn.Dropout(dropout)

        nn.init.orthogonal_(self.proj_scaf.weight)
        nn.init.orthogonal_(self.proj_phys.weight)

    def forward(self, hg: torch.Tensor):
        """
        hg     : (B, dim_graph)
        return : h_scaf, h_phys
        """
        h_scaf = self.dropout(self.norm_scaf(F.gelu(self.proj_scaf(hg))))
        h_phys = self.dropout(self.norm_phys(F.gelu(self.proj_phys(hg))))
        return h_scaf, h_phys

    @staticmethod
    def orth_loss(h_scaf: torch.Tensor, h_phys: torch.Tensor) -> torch.Tensor:
        """
        L_orth = E[ cos^2(h_scaf, h_phys) ]
        """
        h_s = F.normalize(h_scaf, dim=-1)
        h_p = F.normalize(h_phys, dim=-1)
        cos_sim = (h_s * h_p).sum(dim=-1)
        return cos_sim.pow(2).mean()


# =========================================================
# 5. Scaffold Heads
# =========================================================
class ScaffoldAuxHead(nn.Module):
    """
    h_scaf -> scaffold classification
    """
    def __init__(self, dim_graph: int, num_scaffolds: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_graph, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scaffolds)
        )

    def forward(self, h_scaf: torch.Tensor):
        return self.mlp(h_scaf)  # (B, num_scaffolds)


class ScaffoldAdvHead(nn.Module):
    """
    h_phys -(GRL)-> scaffold classification
    """
    def __init__(self, dim_graph: int, num_scaffolds: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_graph, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scaffolds)
        )

    def forward(self, h_phys_reversed: torch.Tensor):
        return self.mlp(h_phys_reversed)  # (B, num_scaffolds)


# =========================================================
# 6. Bilinear Attention
#    q = h_phys
#    k,v = descriptor tokens
# =========================================================
class BilinearAttention(nn.Module):
    def __init__(self, d_q: int, d_t: int, d_attn: int = 64, dropout: float = 0.1):
        super().__init__()
        self.Wq = nn.Linear(d_q, d_attn, bias=False)
        self.Wk = nn.Linear(d_t, d_attn, bias=False)
        self.Wv = nn.Linear(d_t, d_attn, bias=False)

        # bilinear score parameter
        self.U = nn.Parameter(torch.empty(d_attn, d_attn))
        nn.init.xavier_uniform_(self.U)

        self.out_proj = nn.Linear(d_attn, d_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_vec: torch.Tensor, desc_tokens: torch.Tensor, desc_mask: torch.Tensor = None):
        """
        q_vec       : (B, d_q)
        desc_tokens : (B, M, d_t)
        desc_mask   : (B, M), 1 for valid / 0 for invalid (optional)

        return:
            fused      : (B, d_q)
            attn       : (B, M)
        """
        Q = self.Wq(q_vec)              # (B, d_attn)
        K = self.Wk(desc_tokens)        # (B, M, d_attn)
        V = self.Wv(desc_tokens)        # (B, M, d_attn)

        # bilinear score: score_j = Q^T U K_j
        QU = torch.matmul(Q, self.U)                    # (B, d_attn)
        scores = torch.einsum("bd,bmd->bm", QU, K)      # (B, M)

        if desc_mask is not None:
            scores = scores.masked_fill(desc_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)                # (B, M)
        attn = self.dropout(attn)

        context = torch.einsum("bm,bmd->bd", attn, V)   # (B, d_attn)
        fused = self.out_proj(context) + q_vec          # residual to preserve query
        return fused, attn


# =========================================================
# 7. Full Network
# =========================================================
class Net(nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_desc: int,
        num_scaffolds: int = None,
        dim_graph_hidden: int = 100,
        dim_graph: int = 64,
        d_t: int = 32,
        d_attn: int = 64,
        fc1: int = 128,
        fc2: int = 32,
        dropout: float = 0.1,
        grl_lambda: float = 1.0,
    ):
        super().__init__()

        self.num_scaffolds = num_scaffolds
        self.use_scaffold_head = num_scaffolds is not None
        self.grl_lambda = grl_lambda
        self.dim_graph = dim_graph

        # ----- graph encoder -----
        self.gc1 = GCNConv(dim_in, dim_graph_hidden)
        self.gc2 = GCNConv(dim_graph_hidden, dim_graph)

        # ----- descriptor tokenizer -----
        self.desc_tok = DescriptorTokenizer(num_desc=num_desc, d_t=d_t)

        # ----- split graph representation -----
        self.dual_path = DualPathEncoder(dim_graph=dim_graph, dropout=dropout)

        # ----- scaffold heads -----
        if self.use_scaffold_head:
            self.scaf_aux_head = ScaffoldAuxHead(
                dim_graph=dim_graph,
                num_scaffolds=num_scaffolds,
                hidden_dim=dim_graph,
                dropout=dropout
            )
            self.grl = GRL(grl_lambda)
            self.scaf_adv_head = ScaffoldAdvHead(
                dim_graph=dim_graph,
                num_scaffolds=num_scaffolds,
                hidden_dim=dim_graph,
                dropout=dropout
            )
        else:
            self.scaf_aux_head = None
            self.scaf_adv_head = None
            self.grl = None

        # ----- bilinear attention -----
        self.bilinear_attn = BilinearAttention(
            d_q=dim_graph,
            d_t=d_t,
            d_attn=d_attn,
            dropout=dropout
        )

        # ----- prediction head -----
        self.bn1 = nn.BatchNorm1d(fc1)
        self.bn2 = nn.BatchNorm1d(fc2)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(dim_graph, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, 1)

    def encode_graph(self, g):
        """
        return:
            hg: (B, dim_graph)
        """
        x = g.ndata["feat"]
        h = F.relu(self.gc1(g, x))
        h = F.relu(self.gc2(g, h))
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")   # (B, dim_graph)
        return hg

    def forward(self, g, feat_desc, desc_mask=None):
        """
        Args:
            g         : batched DGLGraph
            feat_desc : (B, M)
            desc_mask : (B, M), optional

        Returns:
            dict with keys:
                pred
                hg
                h_scaf
                h_phys
                scaffold_logits_aux
                scaffold_logits_adv
                fused
                attn
        """
        # ----- graph encoding -----
        hg = self.encode_graph(g)                   # (B, dim_graph)

        # ----- descriptor tokenization -----
        desc_tokens = self.desc_tok(feat_desc)      # (B, M, d_t)

        # ----- split representation -----
        h_scaf, h_phys = self.dual_path(hg)         # (B, dim_graph), (B, dim_graph)

        # ----- optional scaffold heads -----
        scaffold_logits_aux = None
        scaffold_logits_adv = None

        if self.use_scaffold_head:
            # auxiliary scaffold prediction on scaffold-specific branch
            scaffold_logits_aux = self.scaf_aux_head(h_scaf)

            # adversarial scaffold prediction on physics/property branch
            h_phys_rev = self.grl(h_phys)
            scaffold_logits_adv = self.scaf_adv_head(h_phys_rev)

        # ----- bilinear attention -----
        # only h_phys is used as query
        fused, attn = self.bilinear_attn(
            q_vec=h_phys,
            desc_tokens=desc_tokens,
            desc_mask=desc_mask
        )  # fused: (B, dim_graph), attn: (B, M)

        # ----- regression head -----
        x = self.fc1(fused)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        pred = self.fc3(x)                          # (B, 1)

        return {
            "pred": pred,
            "hg": hg,
            "h_scaf": h_scaf,
            "h_phys": h_phys,
            "scaffold_logits_aux": scaffold_logits_aux,
            "scaffold_logits_adv": scaffold_logits_adv,
            "fused": fused,
            "attn": attn,
        }
# =========================================================
# 8. Attention Consistency Loss
# =========================================================
def attention_consistency_loss(attn: torch.Tensor, scaffold_ids: torch.Tensor, targets: torch.Tensor = None):
    if scaffold_ids is None:
        return attn.new_tensor(0.0)
    
    B = attn.size(0)
    if B <= 1:
        return attn.new_tensor(0.0)

    attn = F.normalize(attn, dim=-1)

    loss_list = []
    for i in range(B):
        for j in range(i + 1, B):
            # different scaffold only
            if scaffold_ids[i] == scaffold_ids[j]:
                continue

            # if targets exist, only compare samples with similar target
            if targets is not None:
                ti = targets[i].view(-1)[0]
                tj = targets[j].view(-1)[0]
                if torch.abs(ti - tj) > 0.5:  # threshold는 데이터셋에 맞게 조정
                    continue

            sim = (attn[i] * attn[j]).sum()
            loss_list.append(1.0 - sim)

    if len(loss_list) == 0:
        return attn.new_tensor(0.0)

    return torch.stack(loss_list).mean()


# =========================================================
# 9. Total Loss
# =========================================================
def compute_total_loss(
    outputs,
    y,
    scaffold_ids=None,
    criterion_task=None,
    lambda_orth=0.1,
    lambda_aux=0.1,
    lambda_adv=0.1,
    lambda_cons=0.0,
):
    """
    outputs: model(...) dict
    y      : (B, 1)
    scaffold_ids : (B,)
    """
    pred = outputs["pred"]
    h_scaf = outputs["h_scaf"]
    h_phys = outputs["h_phys"]
    scaffold_logits_aux = outputs["scaffold_logits_aux"]
    scaffold_logits_adv = outputs["scaffold_logits_adv"]
    attn = outputs["attn"]

    # 1) task loss
    loss_task = criterion_task(pred, y)

    # 2) orthogonality
    loss_orth = DualPathEncoder.orth_loss(h_scaf, h_phys)

    # 3) default zero losses
    device = pred.device
    loss_aux = torch.tensor(0.0, device=device)
    loss_adv = torch.tensor(0.0, device=device)
    loss_cons = torch.tensor(0.0, device=device)

    # 4) scaffold-related losses only when scaffold labels/logits exist
    use_scaffold_loss = (
        scaffold_ids is not None
        and scaffold_logits_aux is not None
        and scaffold_logits_adv is not None
    )

    if use_scaffold_loss:
        if scaffold_ids.dtype != torch.long:
            scaffold_ids = scaffold_ids.long()

        loss_aux = F.cross_entropy(scaffold_logits_aux, scaffold_ids)
        loss_adv = F.cross_entropy(scaffold_logits_adv, scaffold_ids)
        loss_cons = attention_consistency_loss(attn, scaffold_ids, y)

    total_loss = (
        loss_task
        + lambda_orth * loss_orth
        + lambda_aux * loss_aux
        + lambda_adv * loss_adv
        + lambda_cons * loss_cons
    )

    loss_dict = {
        "loss_total": total_loss.detach(),
        "loss_task": loss_task.detach(),
        "loss_orth": loss_orth.detach(),
        "loss_aux": loss_aux.detach(),
        "loss_adv": loss_adv.detach(),
        "loss_cons": loss_cons.detach(),
    }
    return total_loss, loss_dict