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
    

class CrossAttn_GraphQuery(nn.Module):
    """
    Graph embedding (pooled) as a single Query token.
    Descriptors (2D or 3D) are tokenized as feature-tokens and used as K/V.

    Q:  hg  (B, d_g) -> (B, 1, d_k)
    K,V: x  (B, p)   -> tokenized (B, p, d_g) -> (B, p, d_k)
    Output: hg_attn (B, d_g), attn (B, p)
    """
    def __init__(self, d_g: int, p_desc: int, d_k: int = 32):
        super().__init__()
        self.d_g = d_g
        self.p_desc = p_desc
        self.d_k = d_k

        # feature-token embedding for descriptors: E in R^{p_desc x d_g}
        self.E = nn.Parameter(torch.randn(p_desc, d_g) * 0.02)

        self.Wq = nn.Linear(d_g, d_k, bias=False)
        # self.Wk = nn.Linear(d_g, d_k, bias=False)
        # self.Wv = nn.Linear(d_g, d_k, bias=False)
        self.Wo = nn.Linear(d_k, d_g, bias=False)

        self.Wk = nn.Linear(p_desc, d_k, bias=False)
        self.Wv = nn.Linear(p_desc, d_k, bias=False)

        self.ln = nn.LayerNorm(d_g)

    def forward(self, hg: torch.Tensor, x: torch.Tensor):
        """
        hg: (B, d_g) pooled graph embedding
        x : (B, p_desc) descriptor vector (2D or 3D)
        """
        B, d_g = hg.shape
        # print('hg.shape', hg.shape) # batch x graph embedding dim
        # print('B, d_g', B, d_g) ok
        
        # assert d_g == self.d_g
        # assert x.shape == (B, self.p_desc)

        # Q: (B, 1, d_k)
        Q = self.Wq(hg).unsqueeze(1)
        # print('Q', Q.shape) # batch x 1 x hidden dim

        # Tokenize descriptors: (B, p, d_g)
        T = x.unsqueeze(-1) * self.E.unsqueeze(0)
        # print('x.shape', x.shape) # batch x descriptor dim
        # print('x.unsqueeze(-1)', x.unsqueeze(-1).shape) # batch x descriptor dim x 1

        print('self.E.shape', self.E.shape) # batch x descriptor dim
        # print('E.unsqueeze(0)', self.E.unsqueeze(0).shape) # 1 x descriptor dim x graph emb dim
        # print('T', T.shape) # batch x 1 x graph emb dim

        # K,V: (B, p, d_k)
        K = self.Wk(x).unsqueeze(1)   # (B, 1, d_k)
        V = self.Wv(x).unsqueeze(1)   # (B, 1, d_k)
        # print('K', K.shape) # batch x 1 x hidden dim
        # print('V', V.shape) # batch x 1 x hidden dim

        # K,V: (B, p, d_k)
        # K = self.Wk(T)
        # V = self.Wv(T)
        # print('K', K.shape) # batch x desc x hidden dim
        # print('V', V.shape) # batch x desc x hidden dim

        # Attention: (B, 1, p)
        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        # print('score.shape', scores.shape) # batch x 1 x desc dim

        A = torch.softmax(scores, dim=-1)  # (B, 1, p)
        # print('A.shape', A.shape) # batch x 1 x desc dim

        # Context: (B, 1, d_k) -> (B, d_g)
        Z = torch.matmul(A, V)            # (B, 1, d_k)
        # print('Z.shape', Z.shape) # batch x 1 x hidden dim
        
        Z = self.Wo(Z).squeeze(1)         # (B, d_g)
        # print('2nd Z.shape', Z.shape) # batch x 1 x graph emb dim

        # Residual update
        hg_attn = self.ln(hg + Z)         # (B, d_g)

        return hg_attn, A.squeeze(1)      # (B, p)


class Net(nn.Module):
    """
    hg^(1) = LN(hg + Attn_2d(hg))
    hg^(2) = LN(hg^(1) + Attn_3d(hg^(1)))
    """
    def __init__(self, dim_in, dim_out, dim_self_feat, dim_3d_feat):
        super(Net, self).__init__()

        # Graph encoder
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)
        self.d_g = 20

        # Sequential cross-attention blocks
        self.attn_2d = CrossAttn_GraphQuery(d_g=self.d_g, p_desc=dim_self_feat, d_k=32)
        self.attn_3d = CrossAttn_GraphQuery(d_g=self.d_g, p_desc=dim_3d_feat, d_k=32)

        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, dim_out)

    def forward(self, g, self_feat, x3d, return_attn=False):
        """
        g        : batched DGLGraph
        self_feat: (B, dim_self_feat) 2D descriptor
        x3d      : (B, dim_3d_feat)   3D descriptor
        """
        # ----- 1) Node-level encode
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        # ----- 2) Pool to graph embedding
        hg = dgl.mean_nodes(g, 'h')  # (B, 20)

        # ----- 3) Sequential refinement
        hg1, attn2d = self.attn_2d(hg, self_feat)   # (B,20), (B,dim_self_feat)
        hg2, attn3d = self.attn_3d(hg1, x3d)        # (B,20), (B,dim_3d_feat)

        fused = hg2                  # (B, 20, 1)

        # ----- 5) MLP head
        out = F.relu(self.fc1(fused))
        out = self.fc2(out)

        if return_attn:
            return out, {"attn_2d": attn2d, "attn_3d": attn3d, "hg": hg, "hg1": hg1, "hg2": hg2}
        return out
