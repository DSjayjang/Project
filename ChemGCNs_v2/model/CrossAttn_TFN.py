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


# class Descriptor_Tokenizer(nn.Module):
#     def __init__(self, d_t: int):
#         super().__init__()
#         self.d_t = d_t
#         self.embedding = nn.Linear(1, d_t)

#     def forward(self, descriptor):
#         """
#         descriptor:
#         Tensor of shape (bs, d_desc)

#         return:
#         descriptor token (bs, d_desc, d_t)
#         """
#         # (bs, d_desc) -> (bs, d_desc, 1)
#         desc = descriptor.unsqueeze(-1)

#         # (bs, d_desc, 1) -> (bs, d_desc, d_t)
#         desc_tks = self.embedding(desc)

#         return desc_tks


# class CrossAttn_GraphQuery(nn.Module):
#     """
#     Graph embedding (pooled) as a single Query token.
#     Descriptors (2D or 3D) are tokenized as feature-tokens and used as K/V.

#     Q:  hg  (bs, d_g) -> (bs, 1, d_k)
#     K,V: desc  (bs, d_desc) -> tokenized (bs, d_desc, d_t) -> (bs, d_desc, d_k)
#     Output: hg_attn (bs, d_g), attn (B, d_desc)
#     """
#     def __init__(
#             self, d_g: int, 
#             d_desc: int, 
#             d_t: int = 32, # dim of token
#             d_k: int = 32  # dim of attenion weight
#             ):
#         super().__init__()
#         self.d_g = d_g
#         self.d_desc = d_desc
#         self.d_t = d_t
#         self.d_k = d_k

#         self.desc_tokenizer = Descriptor_Tokenizer(d_t=d_t)

#         self.Wq = nn.Linear(d_g, d_k) # hg -> Q
#         self.Wk = nn.Linear(d_t, d_k) # token -> K
#         self.Wv = nn.Linear(d_t, d_k) # token -> V
#         self.Wo = nn.Linear(d_k, d_g) # context -> delta hg

#         self.ln = nn.LayerNorm(d_g)

#     def forward(self, hg: torch.Tensor, desc: torch.Tensor):
#         """
#         hg(bs, d_g) graph embedding
#         desc: (bs, d_desc) descriptors (2D or 3D)

#         return:
#             hg_attn: (bs, d_g)
#             attn   : (bs, d_desc)   # attention weights
#         """
#         bs, d_g = hg.shape

#         Q = self.Wq(hg).unsqueeze(1) # (bs, 1, d_k)

#         # Tokenization of descriptors: (bs, d_desc, d_t)
#         desc_tks = self.desc_tokenizer(desc)

#         # Key, Value: (bs, d_desc, d_k)
#         K = self.Wk(desc_tks)
#         V = self.Wv(desc_tks)

#         scores = torch.matmul(Q, K.transpose(1,2)) / (self.d_k ** 0.5) # (bs, 1, d_desc)
#         A = torch.softmax(scores, dim=-1) # A: (bs, 1, d_desc)

#         Z = torch.matmul(A, V) # (bs, 1, d_k)
#         Z = self.Wo(Z).squeeze(1) # (bs, 1, d_k) -> (bs, 1, d_g) -> (bs, d_g)

#         hg_attn = self.ln(hg + Z) # (bs, d_g)

#         return hg_attn, A.squeeze(1), Z


# class Net_2d(nn.Module):
#     """
#     hg1 = LN(hg + Attn_2d(hg))
#     hg2 = LN(hg1 + Attn_3d(hg1))
#     """
#     def __init__(self, dim_in, dim_2d_desc, dim_3d_desc):
#         super(Net_2d, self).__init__()

#         dim_out = 1

#         self.dim_2d_desc = dim_2d_desc
#         self.dim_3d_desc = dim_3d_desc

#         # Graph encoder
#         self.dim_graph_emb = 20
#         self.gc1 = GCNConv(dim_in, 100)
#         self.gc2 = GCNConv(100, self.dim_graph_emb)

#         # cross-attention blocks
#         self.attn_2d = CrossAttn_GraphQuery(d_g=self.dim_graph_emb, d_desc=dim_2d_desc)
#         self.attn_3d = CrossAttn_GraphQuery(d_g=self.dim_graph_emb, d_desc=dim_3d_desc)

#         self.fc1 = nn.Linear((self.dim_graph_emb+1) * (self.dim_graph_emb+1), 128)
#         self.fc2 = nn.Linear(128, 32)
#         self.fc3 = nn.Linear(32, dim_out)

#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.bn3 = nn.BatchNorm1d(8)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, g, desc_2d, desc_3d, return_attn=False):
#         """
#         g        : batched DGLGraph
#         desc_2d  : (bs, dim_2d_desc)   2D descriptor
#         desc_3d  : (bs, dim_3d_desc)   3D descriptor
#         """
#         h = F.relu(self.gc1(g, g.ndata['feat']))
#         h = F.relu(self.gc2(g, h))
#         g.ndata['h'] = h

#         hg = dgl.mean_nodes(g, 'h')  # (bs, d_g)
#         batch_size = hg.size(0)

#         # Cross Attention
#         # hg1, hg2: (bs, d_g)
#         hg1, attn2d, z2d = self.attn_2d(hg, desc_2d)  # hg: (bs, d_g), desc_2d: (bs, dim_2d_desc)
#         hg2, attn3d, z3d = self.attn_3d(hg, desc_3d) # hg: (bs, d_g), desc_3d: (bs, dim_3d_desc)

#         # Tensor Fusion
#         ones = torch.ones(batch_size, 1, device=hg.device, dtype=hg.dtype)

#         hg1= torch.cat((hg1, ones), dim = 1) # (bs, d_g+1)
#         z2d = torch.cat((z2d, ones), dim = 1) # (bs, d_g+1)
#         z3d = torch.cat((z3d, ones), dim = 1)

#         # tensor fusion
#         fusion_tensor = torch.bmm(hg1.unsqueeze(2), z2d.unsqueeze(1))
#         fusion_tensor = fusion_tensor.view(batch_size,-1)

#         # MLP
#         out = F.relu(self.bn1(self.fc1(fusion_tensor)))
#         out = self.dropout(out)
#         out = F.relu(self.bn2(self.fc2(out)))
#         out = self.fc3(out)

#         if return_attn:
#             return out, {"attn_2d": attn2d, "attn_3d": attn3d, "hg": hg, "hg1": hg1, "hg2": hg2}
#         return out



class Descriptor_Tokenizer(nn.Module):
    def __init__(self, d_desc:int, d_t: int):
        super().__init__()
        self.val_embedding  = nn.Linear(1, d_t)
        self.id_embedding = nn.Embedding(d_desc, d_t)

    def forward(self, descriptor: torch.Tensor):
        bs, d_desc = descriptor.shape

        desc = descriptor.unsqueeze(-1)     # (bs, d_desc)    -> (bs, d_desc, 1)
        val_emb = self.val_embedding(desc)  # (bs, d_desc, 1) -> (bs, d_desc, d_t)

        # ID embedding
        ids = torch.arange(d_desc, device=descriptor.device)    # (d_desc, )
        id_emb = self.id_embedding(ids)                         # (d_desc, d_t)
        id_emb = id_emb.unsqueeze(0).expand(bs, -1, -1)         # (bs, d_desc, d_t)

        # combine
        desc_tks = val_emb + id_emb

        return desc_tks

class CrossAttn(nn.Module):
    def __init__(self, d_g: int, d_desc: int, d_k: int, d_t: int):
        super().__init__()
        self.d_g = d_g
        self.d_desc = d_desc
        self.d_t = d_t
        self.d_k = d_k

        # Tokenization
        self.desc_tokenizer = Descriptor_Tokenizer(d_desc=d_desc, d_t=d_t)

        # Q, K, V
        self.Wq = nn.Linear(d_g, d_k)
        self.Wk = nn.Linear(d_t, d_k)
        self.Wv = nn.Linear(d_t, d_k)

    def forward(self, hg: torch.Tensor, desc: torch.Tensor):
        # Query
        Q = self.Wq(hg).unsqueeze(1)            # (B, 1, d_k)

        # Key, Value
        desc_tks = self.desc_tokenizer(desc)    # (B, d_desc, d_t)
        K = self.Wk(desc_tks)                   # (B, d_desc, d_k)
        V = self.Wv(desc_tks)                   # (B, d_desc, d_k)

        # scores:  -> attn: (B, d_desc)
        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)     # (B, 1, d_desc)
        attn = torch.softmax(scores, dim=-1)                                # (B, 1, d_desc)

        ctx = torch.matmul(attn, V).squeeze(1)    # (B, d_k)

        return ctx, scores, attn.squeeze(1)


# Cross Attention + Tensor Fusion
class Net_2d(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc, d_t: int=64, d_k: int=32):
        super(Net_2d, self).__init__()

        self.dim_graph_emb = 20

        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_t = d_t
        self.d_k = d_k

        dim_out_fc1 = 128
        dim_out_fc2 = 32
        dim_out = 1
        drop_out = 0.3

        # Graph encoder
        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, self.dim_graph_emb)

        # Cross-Attention blocks
        self.attn_2d = CrossAttn(d_g=self.dim_graph_emb, d_desc=dim_2d_desc, d_t=self.d_t, d_k=self.d_k)
        self.attn_3d = CrossAttn(d_g=self.dim_graph_emb, d_desc=dim_3d_desc, d_t=self.d_t, d_k=self.d_k)

        # MLP blocks
        self.fc1 = nn.Linear((self.dim_graph_emb+1) * (self.d_k+1), dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.BatchNorm1d(dim_out_fc1)
        self.bn2 = nn.BatchNorm1d(dim_out_fc2)

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
        # ctx_3d, scores_3d, attn_3d = self.attn_3d(hg, desc_3d)

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


# TEST!! grid search
class Net_2d_grid(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc, d_t, d_k, dim_out_fc1, dim_out_fc2, drop_out):
        super().__init__()
        self.dim_graph_emb = 20
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_t = d_t
        self.d_k = d_k

        # Graph encoder
        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, self.dim_graph_emb)

        # Cross-Attention blocks
        self.attn_2d = CrossAttn(d_g=self.dim_graph_emb, d_desc=dim_2d_desc, d_t=self.d_t, d_k=self.d_k)
        self.attn_3d = CrossAttn(d_g=self.dim_graph_emb, d_desc=dim_3d_desc, d_t=self.d_t, d_k=self.d_k)

        # MLP blocks
        self.fc1 = nn.Linear((self.dim_graph_emb+1) * (self.d_k+1), dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, 1)

        self.bn1 = nn.BatchNorm1d(dim_out_fc1)
        self.bn2 = nn.BatchNorm1d(dim_out_fc2)

        self.dropout = nn.Dropout(drop_out)

    def forward(self, g, desc_2d, desc_3d):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # (B, d_g)

        B = hg.size(0)
        ctx_2d, _, _ = self.attn_2d(hg, desc_2d)

        # Tensor Fusion
        ones = torch.ones(B, 1, device=hg.device, dtype=hg.dtype)

        tensor_hg = torch.cat((hg, ones), dim=1)            # (B, d_g+1)
        tensor_desc_2d = torch.cat((ctx_2d, ones), dim=1)   # (B, d_k+1)

        fusion_tensor = torch.bmm(tensor_hg.unsqueeze(2), tensor_desc_2d.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(B, -1)

        # MLP
        out = F.relu(self.bn1(self.fc1(fusion_tensor)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)
        
        return out
