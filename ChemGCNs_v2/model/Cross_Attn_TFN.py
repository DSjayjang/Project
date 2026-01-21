import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class NodeApplyModule(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, node):
        h = self.linear(node.data['h'])

        return {'h': h}

class GCNLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCNLayer, self).__init__()
        self.msg = fn.copy_u('h', 'm')
        self.apply_mod = NodeApplyModule(dim_in, dim_out)

    def reduce(self, nodes):
        mbox = nodes.mailbox['m']
        accum = torch.mean(mbox, dim = 1)

        return {'h': accum}     

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(self.msg, self.reduce)
        g.apply_nodes(func = self.apply_mod)

        return g.ndata.pop('h')
    
class Descriptor_Tokenizer(nn.Module):
    def __init__(self, d_t: int):
        super().__init__()
        self.d_t = d_t

        # embedding
        self.embedding = nn.Linear(1, d_t)

    def forward(self, descriptor):
        """
        descriptor:
        Tensor of shape (bs, d_desc)

        return:
        descriptor token (bs, d_desc, d_t)
        """
        bs, d_desc = descriptor.shape

        # (bs, d_desc) -> (bs, d_desc, 1)
        desc = descriptor.unsqueeze(-1)

        # (bs, d_desc, 1) -> (bs, d_desc, d_t)
        desc_tks = self.embedding(desc)

        return desc_tks


class CrossAttn_GraphQuery(nn.Module):
    """
    Graph embedding (pooled) as a single Query token.
    Descriptors (2D or 3D) are tokenized as feature-tokens and used as K/V.

    Q:  hg  (bs, d_g) -> (bs, 1, d_k)
    K,V: desc  (bs, d_desc) -> tokenized (bs, d_desc, d_t) -> (bs, d_desc, d_k)
    Output: hg_attn (bs, d_g), attn (B, d_desc)
    """
    def __init__(
            self, d_g: int, 
            d_desc: int, 
            d_t: int = 32, # dim of token
            d_k: int = 32  # dim of attenion weight
            ):
        super().__init__()
        self.d_g = d_g
        self.d_desc = d_desc
        self.d_t = d_t
        self.d_k = d_k

        self.desc_tokenizer = Descriptor_Tokenizer(d_t=d_t)

        self.Wq = nn.Linear(d_g, d_k) # hg -> Q
        self.Wk = nn.Linear(d_t, d_k) # token -> K
        self.Wv = nn.Linear(d_t, d_k) # token -> V
        self.Wo = nn.Linear(d_k, d_g) # context -> delta hg

        self.ln = nn.LayerNorm(d_g)

    def forward(self, hg: torch.Tensor, desc: torch.Tensor):
        """
        hg(bs, d_g) graph embedding
        desc: (bs, d_desc) descriptors (2D or 3D)

        return:
            hg_attn: (bs, d_g)
            attn   : (bs, d_desc)   # attention weights
        """
        bs, d_g = hg.shape

        Q = self.Wq(hg).unsqueeze(1) # (bs, 1, d_k)

        # Tokenization of descriptors: (bs, d_desc, d_t)
        desc_tks = self.desc_tokenizer(desc)

        # Key, Value: (bs, d_desc, d_k)
        K = self.Wk(desc_tks)
        V = self.Wv(desc_tks)

        scores = torch.matmul(Q, K.transpose(1,2)) / (self.d_k ** 0.5) # (bs, 1, d_desc)
        A = torch.softmax(scores, dim=-1) # A: (bs, 1, d_desc)

        Z = torch.matmul(A, V) # (bs, 1, d_k)
        Z = self.Wo(Z).squeeze(1) # (bs, 1, d_k) -> (bs, 1, d_g) -> (bs, d_g)

        hg_attn = self.ln(hg + Z) # (bs, d_g)

        return hg_attn, A.squeeze(1)


class Net(nn.Module):
    """
    hg1 = LN(hg + Attn_2d(hg))
    hg2 = LN(hg + Attn_3d(hg))
    """
    def __init__(self, dim_in, dim_out, dim_2d_desc, dim_3d_desc):
        super(Net, self).__init__()

        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc

        # Graph encoder
        self.d_g = 20
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, self.d_g)

        # cross-attention blocks
        self.attn_2d = CrossAttn_GraphQuery(d_g=self.d_g, d_desc=dim_2d_desc)
        self.attn_3d = CrossAttn_GraphQuery(d_g=self.d_g, d_desc=dim_3d_desc)

        self.fc1 = nn.Linear((self.d_g+1) * (self.d_g+1) * (self.d_g+1), 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, desc_2d, desc_3d, return_attn=False):
        """
        g        : batched DGLGraph
        desc_2d  : (bs, dim_2d_desc)   2D descriptor
        desc_3d  : (bs, dim_3d_desc)   3D descriptor
        """
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')  # (bs, d_g)
        batch_size = hg.size(0)

        # Cross Attention
        # hg1, hg2: (bs, d_g)
        hg1, attn2d = self.attn_2d(hg, desc_2d)  # hg: (bs, d_g), desc_2d: (bs, dim_2d_desc)
        hg2, attn3d = self.attn_3d(hg, desc_3d) # hg: (bs, d_g), desc_3d: (bs, dim_3d_desc)

        # Tensor Fusion
        ones = torch.ones(batch_size, 1, device=hg.device, dtype=hg.dtype)

        hg = torch.cat((hg, ones), dim = 1)
        hg1 = torch.cat((hg1, ones), dim = 1)
        hg2 = torch.cat((hg2, ones), dim = 1)

        # tensor fusion
        fusion_tensor = torch.einsum('bi, bj, bk -> bijk', hg, hg1, hg2) # (bs, 21, 21, 21)
        fusion_tensor = fusion_tensor.view(batch_size, -1)

        # MLP
        out = F.relu(self.bn1(self.fc1(fusion_tensor)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        if return_attn:
            return out, {"attn_2d": attn2d, "attn_3d": attn3d, "hg": hg, "hg1": hg1, "hg2": hg2}
        return out
