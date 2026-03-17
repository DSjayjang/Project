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

class LowRankFusion(nn.Module):
    def __init__(self, d_g, d_d, d_h, rank):
        super().__init__()
        self.rank = rank
        self.d_h = d_h

        # (rank, d+1, d_out)
        self.w_g = nn.Parameter(torch.empty(rank, d_g+1, d_h))
        self.w_d = nn.Parameter(torch.empty(rank, d_d+1, d_h))
        # nn.init.normal_(self.w_g, std=0.02)
        # nn.init.normal_(self.w_d, std=0.02)
        nn.init.xavier_uniform_(self.w_g)
        nn.init.xavier_uniform_(self.w_d)

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
    
class Net_2d(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc, 
                 d_h: int = 128, rank: int = 8,):
        super(Net_2d, self).__init__()
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_h = d_h

        dim_graph_emb = 20
        dim_out_fc1 = 128
        dim_out_fc2 = 32
        dim_out = 1
        drop_out = 0.3

        # Graph encoder
        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, dim_graph_emb)

        self.low_rank_fusion = LowRankFusion(d_g=dim_graph_emb, d_d=dim_2d_desc, d_h=d_h, rank=rank)
        
        # MLP head
        self.fc1 = nn.Linear(d_h, dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.LayerNorm(dim_out_fc1)
        self.bn2 = nn.LayerNorm(dim_out_fc2)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, g, desc_2d, desc_3d):
        B = g.batch_size

        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # low rank fusion
        final = self.low_rank_fusion(hg, desc_2d)

        # prediction head
        out = F.relu(self.bn1(self.fc1(final)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out

class Net_3d(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc, 
                 d_h: int = 128, rank: int = 8,):
        super(Net_3d, self).__init__()
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_h = d_h

        dim_graph_emb = 20
        dim_out_fc1 = 128
        dim_out_fc2 = 32
        dim_out = 1
        drop_out = 0.3

        # Graph encoder
        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, dim_graph_emb)

        self.low_rank_fusion = LowRankFusion(d_g=dim_graph_emb, d_d=dim_3d_desc, d_h=d_h, rank=rank)
        
        # MLP head
        self.fc1 = nn.Linear(d_h, dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.LayerNorm(dim_out_fc1)
        self.bn2 = nn.LayerNorm(dim_out_fc2)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, g, desc_2d, desc_3d):
        B = g.batch_size

        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # low rank fusion
        final = self.low_rank_fusion(hg, desc_3d)

        # prediction head
        out = F.relu(self.bn1(self.fc1(final)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out


class Net_total(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc, 
                 d_h: int = 64, rank: int = 8,):
        super(Net_total, self).__init__()
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        self.d_h = d_h

        dim_graph_emb = 20
        dim_out_fc1 = 256
        dim_out_fc2 = 64
        dim_out = 1
        drop_out = 0.3

        # Graph encoder
        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, dim_graph_emb)

        self.low_rank_fusion = LowRankFusion(d_g=dim_graph_emb, d_d=dim_2d_desc+dim_3d_desc, d_h=d_h, rank=rank)
        
        # MLP head
        self.fc1 = nn.Linear(d_h, dim_out_fc1)
        self.fc2 = nn.Linear(dim_out_fc1, dim_out_fc2)
        self.fc3 = nn.Linear(dim_out_fc2, dim_out)

        self.bn1 = nn.LayerNorm(dim_out_fc1)
        self.bn2 = nn.LayerNorm(dim_out_fc2)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, g, desc_2d, desc_3d):
        B = g.batch_size

        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # low rank fusion
        concat = torch.cat([desc_2d, desc_3d], dim=1)
        final = self.low_rank_fusion(hg, concat)

        # prediction head
        out = F.relu(self.bn1(self.fc1(final)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out
    