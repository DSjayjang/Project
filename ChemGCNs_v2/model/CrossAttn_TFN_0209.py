import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False):
        super(GCNConv, self).__init__()

        self.weight = nn.Parameter(torch.empty(dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out)) if bias else None
        self.eps = 1e-12
        nn.init.xavier_uniform_(self.weight)

    def forward(self, g, feat: torch.Tensor):
        A_tilde = g.adj().to_dense()
        deg = A_tilde.sum(dim=1)
        D_tilde = torch.pow(deg.clamp_min(self.eps), -0.5)

        X1 = feat * D_tilde.unsqueeze(-1)
        X2 = A_tilde @ X1
        X3 = X2 * D_tilde.unsqueeze(-1)

        out = X3 @ self.weight

        return out        

class Descriptor_Tokenizer(nn.Module):
    def __init__(self, d_desc:int, d_t: int):
        super().__init__()
        # self.val_embedding  = nn.Linear(1, d_t)
        self.id_embedding = nn.Embedding(d_desc, d_t)
        self.val_embedding = nn.Sequential(
                    nn.Linear(1, d_t // 2),
                    nn.ReLU(),
                    nn.Linear(d_t // 2, d_t)
                )
        self.norm = nn.LayerNorm(d_t) # 토큰화 후 정규화 추가

    def forward(self, descriptor: torch.Tensor):
        bs, d_desc = descriptor.shape

        desc = descriptor.unsqueeze(-1)     # (bs, d_desc)    -> (bs, d_desc, 1)
        val_emb = self.val_embedding(desc)  # (bs, d_desc, 1) -> (bs, d_desc, d_t)

        # ID embedding
        ids = torch.arange(d_desc, device=descriptor.device)    # (d_desc, )
        id_emb = self.id_embedding(ids)                         # (d_desc, d_t)
        id_emb = id_emb.unsqueeze(0).expand(bs, -1, -1)         # (bs, d_desc, d_t)

        # combine
        # desc_tks = val_emb + id_emb
        desc_tks = self.norm(val_emb + id_emb)
        return desc_tks

# skip connection 
class CrossAttn(nn.Module):
    def __init__(self, d_g: int, d_desc: int, d_k: int, d_t: int):
        super().__init__()
        self.d_g = d_g
        self.d_desc = d_desc
        self.d_t = d_t
        self.d_k = d_k

        self.use_prenorm = False
        self.ln_hg = nn.LayerNorm(d_g)
        
        # Tokenization
        self.desc_tokenizer = Descriptor_Tokenizer(d_desc=d_desc, d_t=d_t)

        # Q, K, V
        self.Wq = nn.Linear(d_g, d_k)
        self.Wk = nn.Linear(d_t, d_k)
        self.Wv = nn.Linear(d_t, d_k)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)

        # 그래프임베딩을 d_k로
        self.proj_hg = nn.Linear(d_g, d_k)
        self.ctx_drop = nn.Dropout(0.1)
        self.attn_drop = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(d_k)
        self.gate = nn.Sequential(
            nn.Linear(d_g+d_k, d_k),
            nn.ReLU(),
            nn.Linear(d_k, d_k),
            nn.Sigmoid()
        )

    def forward(self, hg: torch.Tensor, desc: torch.Tensor):
        if self.use_prenorm:
            hg = self.ln_hg(hg)
        else:
            hg = hg

        # Query
        Q = self.Wq(hg).unsqueeze(1)            # (B, 1, d_k)

        # Key, Value
        desc_tks = self.desc_tokenizer(desc)    # (B, d_desc, d_t)
        K = self.Wk(desc_tks)                   # (B, d_desc, d_k)
        V = self.Wv(desc_tks)                   # (B, d_desc, d_k)

        # scores:  -> attn: (B, d_desc)
        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)     # (B, 1, d_desc)
        attn = torch.softmax(scores, dim=-1)                                # (B, 1, d_desc)

        attn = self.attn_drop(attn)
        
        ctx = torch.matmul(attn, V).squeeze(1)    # (B, d_k)

        # residual learning
        base = self.proj_hg(hg)
        g = self.gate(torch.cat([hg, ctx], dim=-1))  # (B,d_k)
        out = base + g*(self.ctx_drop(ctx)-base)

        # out = base +ctx

        out = self.ln(out)

        return out, scores, attn.squeeze(1)


# Cross Attention + Tensor Fusion
# 멀티헤드에서는 d_t:32, d_k:64, fc1 256:, fc2:64, head:8, dropour:0.1
# freesolv 0.901, 0.857, esol 0.909, -0.001
class Net_2d(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc, d_t: int=32, d_k: int=64):
        super(Net_2d, self).__init__()

        self.dim_graph_emb = 20

        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_t = d_t
        self.d_k = d_k

        dim_out_fc1 = 256
        dim_out_fc2 = 64
        dim_out = 1
        drop_out = 0.1

        # Graph encoder
        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, self.dim_graph_emb)

        # Cross-Attention blocks
        # self.attn_2d = CrossAttn(d_g=self.dim_graph_emb, d_desc=dim_2d_desc, d_t=self.d_t, d_k=self.d_k)
        # self.attn_3d = CrossAttn(d_g=self.dim_graph_emb, d_desc=dim_3d_desc, d_t=self.d_t, d_k=self.d_k)
        self.attn_2d = MultiHeadCrossAttn(d_g=self.dim_graph_emb, d_desc=dim_2d_desc, d_t=self.d_t, d_k=self.d_k)
        self.attn_3d = MultiHeadCrossAttn(d_g=self.dim_graph_emb, d_desc=dim_3d_desc, d_t=self.d_t, d_k=self.d_k)

        # MLP blocks
        self.fc1 = nn.Linear((self.dim_graph_emb+1) * (self.d_k+1), dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.LayerNorm(dim_out_fc1)
        self.bn2 = nn.LayerNorm(dim_out_fc2)

        self.dropout = nn.Dropout(drop_out)


    def forward(self, g, desc_2d, desc_3d, return_attn=False):
        # Graph embedding
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # (B, d_g)
        
        B = hg.size(0)

        # Cross-attention
        ctx_2d, scores_2d, attn_2d = self.attn_2d(hg, desc_2d)

        # Tensor Fusion
        ones = torch.ones(B, 1, device=hg.device, dtype=hg.dtype)

        tensor_hg = torch.cat((hg, ones), dim=1)            # (B, d_g+1)
        tensor_desc_2d = torch.cat((ctx_2d, ones), dim=1)   # (B, d_k+1)

        fusion_tensor = torch.bmm(tensor_hg.unsqueeze(2), tensor_desc_2d.unsqueeze(1))  # (B, d_g+1, d_k+1)
        fusion_tensor = fusion_tensor.view(B, -1)                                       # (B, (d_g+1)*(d_k+1))

        # MLP
        out = F.relu(self.bn1(self.fc1(fusion_tensor)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out

# 0.906
class MultiHeadCrossAttn(nn.Module):
    def __init__(
        self,
        d_g: int,
        d_desc: int,
        d_t: int,
        d_k: int,
        num_heads: int = 8,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        use_residual: bool = True,
        use_prenorm: bool = True,
    ):
        super().__init__()
        self.d_g = d_g
        self.d_desc = d_desc
        self.d_t = d_t
        self.d_k = d_k
        self.H = num_heads
        self.d_h = d_k // num_heads

        # Tokenization
        self.desc_tokenizer = Descriptor_Tokenizer(d_desc=d_desc, d_t=d_t)

        # (선택) Pre-Norm
        self.use_prenorm = use_prenorm
        if use_prenorm:
            self.ln_hg = nn.LayerNorm(d_g)
            self.ln_tk = nn.LayerNorm(d_t)

        # QKV projections
        self.Wq = nn.Linear(d_g, d_k)
        self.Wk = nn.Linear(d_t, d_k)
        self.Wv = nn.Linear(d_t, d_k)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)

        # output projection
        self.Wo = nn.Linear(d_k, d_k)

        # dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # residual stabilization
        self.use_residual = use_residual
        if use_residual:
            self.proj_hg = nn.Linear(d_g, d_k)
            self.ln_out = nn.LayerNorm(d_k)
        self.gate = nn.Sequential(
            nn.Linear(d_g+d_k, d_k),
            nn.ReLU(),
            nn.Linear(d_k, d_k),
            nn.Sigmoid()
        )

    def forward(self, hg: torch.Tensor, desc: torch.Tensor):
        B, D = desc.shape

        if self.use_prenorm:
            hg_in = self.ln_hg(hg)
        else:
            hg_in = hg

        # descriptor tokens
        desc_tks = self.desc_tokenizer(desc)  # (B, D, d_t)
        if self.use_prenorm:
            desc_tks = self.ln_tk(desc_tks)

        # project Q,K,V to d_k then split heads
        Q = self.Wq(hg_in)                    # (B, d_k)
        K = self.Wk(desc_tks)                 # (B, D, d_k)
        V = self.Wv(desc_tks)                 # (B, D, d_k)

        # reshape to multi-head:
        # Q: (B, H, 1, d_h), K/V: (B, H, D, d_h)
        Q = Q.reshape(B, self.H, self.d_h).unsqueeze(2)             # (B,H,1,d_h)
        K = K.reshape(B, D, self.H, self.d_h).transpose(1, 2)       # (B,H,D,d_h)
        V = V.reshape(B, D, self.H, self.d_h).transpose(1, 2)       # (B,H,D,d_h)

        # scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_h ** 0.5)   # (B,H,1,D)
        attn = torch.softmax(scores, dim=-1)                                # (B,H,1,D)
        attn = self.attn_drop(attn)

        ctx = torch.matmul(attn, V).squeeze(2)                              # (B,H,d_h)
        ctx = ctx.reshape(B, self.d_k)                                      # (B,d_k)
        ctx = self.Wo(ctx)
        ctx = self.proj_drop(ctx)

        # residual (optional)
        if self.use_residual:
            base = self.proj_hg(hg)                                         # (B,d_k)
            out = self.ln_out(base + ctx)
        else:
            out = ctx

        return out, scores.squeeze(2), attn.squeeze(2)  # (B,d_k), (B,H,D), (B,H,D)

