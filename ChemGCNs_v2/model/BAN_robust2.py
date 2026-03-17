import torch
import torch.nn as nn
# from torch.nn.utils import weight_norm
import math
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

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GRL.apply(x, lambd)

class ScaffoldAdversary(nn.Module):
    def __init__(self, dim_in, num_scaffolds, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_scaffolds)
        )

    def forward(self, z, lambd=1.0):
        z_rev = grad_reverse(z, lambd)
        return self.net(z_rev)


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

class BilinearAttention(nn.Module):
    def __init__(self, d_q: int, d_t: int, glimpse: int, K: int=None, bias=False):
        super().__init__()

        self.d_q = d_q
        self.d_t = d_t
        self.glimpse = glimpse
        
        self.K_joint = d_t if K is None else K
        self.K_attn  = self.K_joint

        self.U_attn = nn.Linear(d_q, self.K_attn, bias=bias)
        self.V_attn = nn.Linear(d_t, self.K_attn, bias=bias)
        self.p = nn.Parameter(torch.empty(self.glimpse, self.K_attn))
        nn.init.normal_(self.p, mean=0, std=1.0 / math.sqrt(self.K_attn))

        self.U_joint = nn.Linear(d_q, self.K_joint, bias=bias)  # U' : d_q → K
        self.V_joint = nn.Linear(d_t, self.K_joint, bias=bias)  # V' : d_t → K
        self.P = nn.Linear(self.K_joint, d_q, bias=bias)        # P  : K   → C(=d_q)

        self.res_norm = nn.ModuleList([nn.LayerNorm(d_q) for _ in range(self.glimpse)])

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """
        X:  (bs, dim_graph)
        Y:  (bs, dim_desc_2d, d_t)
        """
        bs, M, d_t = Y.shape
        f_i = X

        YV_attn   = F.relu(self.V_attn(Y))    # (bs, M, K)
        YV_joint = F.relu(self.V_joint(Y))  # (bs, M, K)

        attn_list = []

        for g in range(self.glimpse):
            XU_attn = F.relu(self.U_attn(f_i))               # (bs, K)
            logits = torch.einsum('bk,bmk,k->bm', XU_attn, YV_attn, self.p[g])  # (bs, M)

            attn = F.softmax(logits, dim=-1)             # (bs, M)
            attn_list.append(attn)

            Vy_hat = torch.einsum('bm,bmk->bk', attn, YV_joint)              # (bs, K)

            XU_joint = F.relu(self.U_joint(f_i))           # (bs, K)
            f_joint  = self.P(XU_joint * Vy_hat)           # (bs, d_q)

            f_i = self.res_norm[g](f_i + f_joint)

        return f_i, attn_list


# class Net(nn.Module):
#     def __init__(self, dim_in: int, dim_desc_2d: int, num_scaffolds: int):
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_desc_2d = dim_desc_2d

#         self.dim_graph = 20    # 20기본
#         self.dim_fc1 = 256      # 256기본
#         self.dim_fc2 = 64
#         self.dim_out = 1
#         self.drop_out = 0.2

#         self.d_t = 64        # 128기본 64최고
#         self.K = 16           # 32기본 16최고
#         self.glimpse = 2      # 4기본 2최고

#         self.gc1 = GCNConv(dim_in, 100)
#         self.gc2 = GCNConv(100, self.dim_graph)

#         self.fc1 = nn.Linear(self.dim_graph, self.dim_fc1)
#         self.fc2 = nn.Linear(self.dim_fc1, self.dim_fc2)
#         self.fc3 = nn.Linear(self.dim_fc2, self.dim_out)

#         self.bn1 = nn.LayerNorm(self.dim_fc1)
#         self.bn2 = nn.LayerNorm(self.dim_fc2)
#         self.dropout = nn.Dropout(self.drop_out)

#         self.tokenizer = Descriptor_Tokenizer(self.dim_desc_2d, self.d_t)
        
#         # Bilinear Cross Attention
#         self.ban = BilinearAttention(d_q=self.dim_graph, d_t=self.d_t, glimpse=self.glimpse, K=self.K)
#         self.scaffold_adv = ScaffoldAdversary(self.dim_graph, num_scaffolds)


#     def forward(self, g, desc_2d):
#         h = F.relu(self.gc1(g, g.ndata['feat']))
#         h = F.relu(self.gc2(g, h))
#         g.ndata['h'] = h
#         hg = dgl.mean_nodes(g, 'h')             # (bs, dim_graph)    

#         # Descriptor tokenizaiton         
#         desc_tks = self.tokenizer(desc_2d)      # (bs, dim_desc_2d, d_t)
#         fused, attn_list = self.ban(hg, desc_tks)
#         scaf_logits = self.scaffold_adv(fused, lambd=0.01)
#         # Bilinear Attention

#         out = F.relu(self.bn1(self.fc1(fused)))
#         out = self.dropout(out)
#         out = F.relu(self.bn2(self.fc2(out)))
#         out = self.fc3(out)

#         return out, attn_list, scaf_logits
    
class Net(nn.Module):
    def __init__(self, dim_in: int, dim_desc_2d: int, num_scaffolds: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_desc_2d = dim_desc_2d

        self.dim_graph = 20
        self.dim_fc1 = 256
        self.dim_fc2 = 64
        self.dim_out = 1
        self.drop_out = 0.2

        self.d_t = 64
        self.K = 16
        self.glimpse = 2

        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, self.dim_graph)

        self.fc1 = nn.Linear(self.dim_graph, self.dim_fc1)
        self.fc2 = nn.Linear(self.dim_fc1, self.dim_fc2)
        self.fc3 = nn.Linear(self.dim_fc2, self.dim_out)

        self.bn1 = nn.LayerNorm(self.dim_fc1)
        self.bn2 = nn.LayerNorm(self.dim_fc2)
        self.dropout = nn.Dropout(self.drop_out)

        self.tokenizer = Descriptor_Tokenizer(self.dim_desc_2d, self.d_t)
        self.ban = BilinearAttention(
            d_q=self.dim_graph, d_t=self.d_t,
            glimpse=self.glimpse, K=self.K
        )
        self.scaffold_adv = ScaffoldAdversary(self.dim_graph, num_scaffolds)

    def forward(self, g, desc_2d, adv_lambda=0.0):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        desc_tks = self.tokenizer(desc_2d)
        fused, attn_list = self.ban(hg, desc_tks)

        scaf_logits = self.scaffold_adv(fused, lambd=adv_lambda)

        out = F.relu(self.bn1(self.fc1(fused)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out, attn_list, scaf_logits
    
# 장점 compared to KROVEX
# parameter efficient
# interpretability to descriptors with attention mechanism
# good when scffold split