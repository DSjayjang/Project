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
        self.id_embedding = nn.Embedding(d_desc, d_t)
        self.val_embedding = nn.Sequential(
                    nn.Linear(1, d_t // 2),
                    nn.ReLU(),
                    nn.Linear(d_t // 2, d_t)
                )
        self.norm = nn.LayerNorm(d_t)

    def forward(self, descriptor: torch.Tensor):
        bs, d_desc = descriptor.shape

        desc = descriptor.unsqueeze(-1)     # (bs, d_desc)    -> (bs, d_desc, 1)
        val_emb = self.val_embedding(desc)  # (bs, d_desc, 1) -> (bs, d_desc, d_t)

        # ID embedding
        ids = torch.arange(d_desc, device=descriptor.device)    # (d_desc, )
        id_emb = self.id_embedding(ids)                         # (d_desc, d_t)
        id_emb = id_emb.unsqueeze(0).expand(bs, -1, -1)         # (bs, d_desc, d_t)

        # combine
        desc_tks = self.norm(val_emb + id_emb)
        return desc_tks


def unpack_nodes(bg, h):
    """
    bg: batched DGLGraph
    h : (Total umber of nodes in a batch, dim_graph_emb)

    return:
      H: (B, Nmax, d_g)
      node_pad_mask: (B, Nmax)  True=pad, False=valid
    """
    num_nodes = bg.batch_num_nodes().tolist() 
    # num_nodes e.g., [3, 5, 6, 7, 5, 9, 6, 7, 22, 6, 7, 11, 2, 6, 7, 17, 7, 4, 23, 8, 7, 7, 4, 3, 9, 10, 8, 6, 9, 10, 7, 3] 

    B = len(num_nodes) # len of batch size
    d_g = h.size(1)
    Nmax = max(num_nodes)
    H = h.new_zeros((B, Nmax, d_g)) # (bs, Nmax, d_g)
    node_pad_mask = torch.ones((B, Nmax), dtype=torch.bool, device=h.device) # (bs, Nmax)

    s = 0
    for i, n in enumerate(num_nodes):
        H[i, :n] = h[s:s+n]
        node_pad_mask[i, :n] = False
        s += n

    return H, node_pad_mask


class MultiHeadCrossAttn(nn.Module):
    """
    Bidirectional Cross-Attention:
      1) nodes (Q) attend to desc (K,V) -> g_vec
      2) desc  (Q) attend to nodes(K,V) -> d_vec
    """
    def __init__(
            self,
            d_g: int, 
            d_desc: int, 
            d_t: int,
            d_k: int,
            num_heads: int = 4,
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            use_residual: bool = True,
            use_prenorm: bool = True
        ):
        super(MultiHeadCrossAttn, self).__init__()
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
            self.ln_node = nn.LayerNorm(d_g)
            self.ln_desc = nn.LayerNorm(d_t)

        # ===== 방향 1: nodes(Q) <- desc(K,V) =====
        self.Wq_g = nn.Linear(d_g, d_k)
        self.Wk_g = nn.Linear(d_t, d_k)
        self.Wv_g = nn.Linear(d_t, d_k)

        # ===== 방향 2: desc(Q) <- nodes(K,V) =====
        self.Wq_d = nn.Linear(d_t, d_k)
        self.Wk_d = nn.Linear(d_g, d_k)
        self.Wv_d = nn.Linear(d_g, d_k)

        # output projections
        self.Wo_g = nn.Linear(d_k, d_k)
        self.Wo_d = nn.Linear(d_k, d_k)

        # dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # residual + gating
        self.use_residual = use_residual
        if use_residual:
            self.proj_node = nn.Linear(d_g, d_k)  # nodes residual base
            self.proj_desc = nn.Linear(d_t, d_k)  # desc residual base
            self.ln_out_g = nn.LayerNorm(d_k)
            self.ln_out_d = nn.LayerNorm(d_k)

        # gates (token-wise gate)
        self.gate_g = nn.Sequential(
            nn.Linear(d_g + d_k, d_k),
            nn.ReLU(),
            nn.Linear(d_k, d_k),
            nn.Sigmoid()
        )
        self.gate_d = nn.Sequential(
            nn.Linear(d_t + d_k, d_k),
            nn.ReLU(),
            nn.Linear(d_k, d_k),
            nn.Sigmoid()
        )
         # init
        for m in [self.Wq_g, self.Wk_g, self.Wv_g, self.Wq_d, self.Wk_d, self.Wv_d, self.Wo_g, self.Wo_d]:
            nn.init.xavier_uniform_(m.weight)


    def _mh_attn(self, Q, K, V, key_pad_mask=None):
        """
        Q: (B, Tq, d_k)
        K: (B, Tk, d_k)
        V: (B, Tk, d_k)
        key_pad_mask: (B, Tk) True=pad
        return:
          ctx: (B, Tq, d_k)
          scores: (B, H, Tq, Tk)  (mask 적용 전 softmax 전 score)
          attn:   (B, H, Tq, Tk)
        """
        B, Tq, _ = Q.shape
        Tk = K.shape[1]

        # split heads
        Qh = Q.reshape(B, Tq, self.H, self.d_h).transpose(1, 2)  # (B,H,Tq,d_h)
        Kh = K.reshape(B, Tk, self.H, self.d_h).transpose(1, 2)  # (B,H,Tk,d_h)
        Vh = V.reshape(B, Tk, self.H, self.d_h).transpose(1, 2)  # (B,H,Tk,d_h)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.d_h ** 0.5)  # (B,H,Tq,Tk)

        # key padding mask 적용: pad 위치는 -inf로
        if key_pad_mask is not None:
            # key_pad_mask: (B,Tk) -> (B,1,1,Tk)
            scores = scores.masked_fill(key_pad_mask[:, None, None, :], float('-inf'))

        attn = torch.softmax(scores, dim=-1)                  # (B,H,Tq,Tk)
        # attn = self.attn_drop(attn)

        ctx = torch.matmul(attn, Vh)                          # (B,H,Tq,d_h)
        ctx = ctx.transpose(1, 2).contiguous().reshape(B, Tq, self.d_k)  # (B,Tq,d_k)

        return ctx, scores, attn

    @staticmethod
    def _masked_mean(tokens, pad_mask):
        """
        tokens: (B,T,d)
        pad_mask: (B,T) True=pad
        return: (B,d)
        """
        valid = (~pad_mask).unsqueeze(-1).float()  # (B,T,1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (tokens * valid).sum(dim=1) / denom

    def forward(self, node_tokens, desc, node_pad_mask=None):
        """
        node_tokens: (B, N, d_g)  (GCN 이후 노드 토큰)
        desc:        (B, D)       (raw descriptor vector)
        node_pad_mask: (B, N) True=pad (없으면 None)
        return:
          g_vec: (B,d_k)   # nodes-as-query 결과를 pooling한 graph-side vector
          d_vec: (B,d_k)   # desc-as-query 결과를 pooling한 desc-side vector
          aux: dict(attn maps...)
        """
        B, N, _ = node_tokens.shape
        _, D = desc.shape

        # prenorm
        if self.use_prenorm:
            node_in = self.ln_node(node_tokens)
        else:
            node_in = node_tokens

        # desc tokens
        desc_tks = self.desc_tokenizer(desc)  # (B,D,d_t)
        if self.use_prenorm:
            desc_in = self.ln_desc(desc_tks)
        else:
            desc_in = desc_tks

        # ===== 방향 1: nodes(Q) <- desc(K,V) =====
        Qg = self.Wq_g(node_in)     # (B,N,d_k)
        Kg = self.Wk_g(desc_in)     # (B,D,d_k)
        Vg = self.Wv_g(desc_in)     # (B,D,d_k)

        ctx_g, scores_g, attn_g = self._mh_attn(Qg, Kg, Vg, key_pad_mask=None)  # desc는 고정길이라 mask 보통 None
        # ctx_g = self.proj_drop(ctx_g)  # (B,N,d_k)
        # ctx_g = self.proj_drop(self.Wo_g(ctx_g))  # (B,N,d_k)

        # gate_g = self.gate_g(torch.cat([node_tokens, ctx_g], dim=-1))  # (B,N,d_k)
        # ctx_g = gate_g * ctx_g

        if self.use_residual:
            base_g = self.proj_node(node_tokens)  # (B,N,d_k)
            out_g_tokens = self.ln_out_g(base_g + ctx_g)
        else:
            out_g_tokens = ctx_g

        # pool -> g_vec
        if node_pad_mask is None:
            g_vec = out_g_tokens.mean(dim=1)  # (B,d_k)
        else:
            g_vec = self._masked_mean(out_g_tokens, node_pad_mask)

        # ===== 방향 2: desc(Q) <- nodes(K,V) =====
        Qd = self.Wq_d(desc_in)     # (B,D,d_k)
        Kd = self.Wk_d(node_in)     # (B,N,d_k)
        Vd = self.Wv_d(node_in)     # (B,N,d_k)

        ctx_d, scores_d, attn_d = self._mh_attn(Qd, Kd, Vd, key_pad_mask=node_pad_mask)  # node pad mask 적용
        # ctx_d = self.proj_drop(ctx_d)  # (B,D,d_k)
        # ctx_d = self.proj_drop(self.Wo_d(ctx_d))  # (B,D,d_k)

        # gate_d = self.gate_d(torch.cat([desc_tks, ctx_d], dim=-1))  # (B,D,d_k)
        # ctx_d = gate_d * ctx_d

        if self.use_residual:
            base_d = self.proj_desc(desc_tks)      # (B,D,d_k)
            out_d_tokens = self.ln_out_d(base_d + ctx_d)
        else:
            out_d_tokens = ctx_d

        # pool -> d_vec (descriptor는 padding 없으니 mean)
        d_vec = out_d_tokens.mean(dim=1)  # (B,d_k)

        aux = {
            "out_g_tokens": out_g_tokens,   # (B,N,d_k)
            "out_d_tokens": out_d_tokens,   # (B,D,d_k)
            "scores_g": scores_g,           # (B,H,N,D)
            "attn_g": attn_g,               # (B,H,N,D)
            "scores_d": scores_d,           # (B,H,D,N)
            "attn_d": attn_d,               # (B,H,D,N)
        }
        return g_vec, d_vec, aux

class LowRankFusion(nn.Module):
    def __init__(self, d_g, d_d, d_h, rank):
        super().__init__()
        self.rank = rank
        self.d_h = d_h

        # (rank, d+1, d_out)
        self.w_g = nn.Parameter(torch.empty(rank, d_g+1, d_h))
        self.w_d = nn.Parameter(torch.empty(rank, d_d+1, d_h))
        nn.init.normal_(self.w_g, std=0.02)
        nn.init.normal_(self.w_d, std=0.02)

    def forward(self, g_vec, d_vec):
        B = g_vec.size(0)
        ones = torch.ones(B, 1, device=g_vec.device, dtype=g_vec.dtype)

        g_h = torch.cat([g_vec, ones], dim=1)   # (B, d_g+1)
        d_h = torch.cat([d_vec, ones], dim=1)   # (B, d_d+1)

        # efficient low-rank
        fusion_g = torch.einsum('bd,rdh->brh', g_h, self.w_g)  # (B, r, d_h)
        fusion_d = torch.einsum('bd,rdh->brh', d_h, self.w_d)  # (B, r, d_h)
        fusion = fusion_g * fusion_d                           # (B, r, d_h)

        res = fusion.sum(dim=1)

        return res
    
class Net(nn.Module):
    def __init__(
            self, 
            dim_in, 
            dim_2d_desc, 
            dim_3d_desc, 
            d_t: int, 
            d_k: int,
            d_h: int,
            dim_out_fc1, 
            dim_out_fc2, 
            drop_out, 
            num_heads,
            rank: int,
            use_pe: bool = True):
        super(Net, self).__init__()
        pos_enc_dim = 8
        self.dim_graph_emb = 64

        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_t = d_t
        self.d_k = d_k

        dim_out = 1


        # LapPE
        self.use_pe = use_pe
        self.pe_proj = nn.Linear(pos_enc_dim, dim_in, bias=True)

        # Graph encoder
        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, self.dim_graph_emb)   # node dim = 20

        # Bidirectional Cross-Attn (nodes<->desc)
        self.cross_2d = MultiHeadCrossAttn(
            d_g=self.dim_graph_emb,     # node token dim = 20
            d_desc=dim_2d_desc,         # number of descriptors (D)
            d_t=d_t,                    # desc token dim
            d_k=d_k,                    # attention model dim
            num_heads=num_heads
        )

        self.rank=rank
        self.low_rank_fusion = LowRankFusion(d_g=d_k, d_d=d_k, d_h=d_h, rank=self.rank)
        # MLP head
        self.fc1 = nn.Linear(d_h, dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.LayerNorm(dim_out_fc1)
        self.bn2 = nn.LayerNorm(dim_out_fc2)

        self.dropout = nn.Dropout(drop_out)

    def forward(self, g, desc_2d, desc_3d):
        B = g.batch_size

        # node features + PE
        x = g.ndata['feat']
        if self.use_pe:
            p = g.ndata['lap_pos_enc']
            x = x + self.pe_proj(p)

        # GCN -> node embeddings (N_total, 20)
        h = F.relu(self.gc1(g, x))
        h = F.relu(self.gc2(g, h))

        # unpack node tokens: (B,Nmax,20) + pad mask
        node_tokens, node_pad_mask = unpack_nodes(g, h)

        # bidirectional cross-attn
        g_vec, d_vec, aux = self.cross_2d(node_tokens, desc_2d, node_pad_mask=node_pad_mask)
        # g_vec: (B,d_k), d_vec: (B,d_k)
        final = self.low_rank_fusion(g_vec, d_vec) # (B, 128)
        # out = F.relu(self.bn1(self.fc1(final)))
        # # ---------------------------
        # # Bilinear Fusion (Low-rank)
        # # ---------------------------
        # g_r = self.bilin_g(g_vec)          # (B, r)
        # d_r = self.bilin_d(d_vec)          # (B, r)
        # fused = self.bilin_ln(g_r * d_r)
        # final = F.relu(fused)
        # # final = self.bilin_drop(F.relu(fused))
        # # fused = g_r * d_r                  # (B, r)  element-wise Hadamard (bilinear 핵심)
        # # fused = F.relu(fused)              # (선택) 비선형
        # # final = fused
        # # final = self.bilin_out(fused)      # (B, 256)  -> MLP 입력 차원으로 맞춤
        # # ---------------------------


        # prediction head
        out = F.relu(self.bn1(self.fc1(final)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        # 필요하면 attention map도 같이 반환 가능
        return out  # 또는 return out, aux
    