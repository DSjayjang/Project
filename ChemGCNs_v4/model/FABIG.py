import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.function as dglfn

from utils.family import family_indices


class GCNConv(nn.Module):
    """
    Graph Convolutional Network layer.
 
    변경 1: adj().to_dense() 제거 → dgl update_all() 기반 message passing
            batched graph에서 분자별 정규화가 올바르게 동작하도록 수정.
 
    변경 2: self-loop (A_hat = A + I) 추가
            자기 자신의 feature가 업데이트에 반영되지 않던 문제 해결.
            dgl.add_self_loop()은 호출 시점(forward)이 아닌 외부에서
            graph 전처리 시 수행하거나, 아래처럼 degree 계산에 반영.
    """
 
    def __init__(self, dim_in, dim_out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim_in, dim_out))
        self.bias   = nn.Parameter(torch.zeros(dim_out)) if bias else None
        self.eps    = 1e-12
        nn.init.xavier_uniform_(self.weight)
 
    def forward(self, g, feat: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            # self-loop을 추가한 그래프에서 degree 계산
            # dgl.add_self_loop은 그래프를 복사하므로 local_scope 안에서 처리
            g_sl = dgl.add_self_loop(g)
 
            # D^{-1/2} 계산 (self-loop 포함 degree 기준)
            deg = g_sl.in_degrees().float().clamp(min=self.eps)
            d_inv_sqrt = deg.pow(-0.5)
 
            # XW 먼저 계산 (선형 변환)
            g_sl.ndata['h'] = feat @ self.weight           # [N, dim_out]
 
            # D^{-1/2} X W
            g_sl.ndata['h'] = g_sl.ndata['h'] * d_inv_sqrt.unsqueeze(-1)
 
            # A * (D^{-1/2} X W): message passing
            g_sl.update_all(
                message_func=dglfn.copy_u('h', 'm'),
                reduce_func=dglfn.sum('m', 'h_agg')
            )
 
            # D^{-1/2} * (A * D^{-1/2} X W)
            out = g_sl.ndata['h_agg'] * d_inv_sqrt.unsqueeze(-1)
 
            if self.bias is not None:
                out = out + self.bias
 
        return out


class FamilyTokenizer(nn.Module):
    """
    Tokenizer
    각 family당 받은 desc를 정규화 후 딕셔너리에 저장

    outputs:
    tokns_dict: dict
    {fam: (bs, dim_desc_hiddn)}
    """
    def __init__(self, family_len: dict, dim_desc_hidden: int, dropout: float):
        super().__init__()

        self.encoders = nn.ModuleDict({
            fam: nn.Sequential(
                nn.Linear(num_desc, dim_desc_hidden),
                nn.LayerNorm(dim_desc_hidden),
                nn.ReLU(),
                nn.Dropout(dropout))
            for fam, num_desc in family_len.items()})

    def forward(self, family_inputs: dict):
        tokns_dict = {}

        for fam, x in family_inputs.items():
            tokns_dict[fam] = self.encoders[fam](x)

        return tokns_dict



def split_pad_node_embeddings(bg, h):
    """
    DGL batched graph의 “붙어있는 노드 임베딩(h)”을 → 그래프별로 분리하고 → 동일 길이로 padding하여 3D 텐서로 변환

    bg: ???
    h: (N_total, dim_graph)
    """

    # 각 배치 당 노드 개수
    # e.g., 
    # num_nodes_list = [5, 14, 8, 8, 4, 2, 5, 3, 7, 11, 8, 7, 8, 8, 10, 17, 5, 4, 5, 16, 3, 5, 12, 11, 8, 10, 4, 9, 10, 7, 9, 6]
    # num_nodes_list = [9, 8, 2, 10, 7, 11, 3, 8, 3, 17, 14, 9, 9, 7, 10, 6, 7, 11, 7, 16, 10, 6, 14, 4, 2, 12, 8, 5, 4, 6, 10, 12]
    num_nodes_list = bg.batch_num_nodes().tolist() # 노드 임베딩을 그래프별로 분리

    B = len(num_nodes_list) # batch size
    d = h.size(-1) # feature dimension = dim_graph
    N_max = max(num_nodes_list)

    H_pad = h.new_zeros(B, N_max, d) # 노드 임베딩을 저장할 공간 # (bs, max, 64)
    mask = torch.zeros(B, N_max, dtype=torch.bool, device=h.device) # (bs, N_max)

    """
    h에서 해당 그래프에 해당하는 노드만 잘라서
    H_pad[b]에 넣음
    나머지는 padding (0)
    """
    start = 0
    for b, n in enumerate(num_nodes_list):
        H_pad[b, :n] = h[start:start+n] # h[start:start+n].shape : (각 그래프의 노드 개수 x dim_graph)
        mask[b, :n] = True
        start += n

    return H_pad, mask


class FamilyBilinearAttention(nn.Module):
    def __init__(self, dim_graph, dim_desc_hidden, rank):
        super().__init__()
        self.q_proj = nn.Linear(dim_desc_hidden, rank, bias=False) # Wq for fam
        self.k_proj = nn.Linear(dim_graph, rank, bias=False) # Wk for graph

        # # 추가: 노드와 디스크립터 특징의 스케일을 맞춰줍니다.
        # self.q_norm = nn.LayerNorm(rank)
        # self.k_norm = nn.LayerNorm(rank)
        
        
        # self.scale = rank ** 0.5
        # attention 후 context를 dim_desc_hidden 투영
        self.v_proj   = nn.Linear(dim_graph, dim_desc_hidden, bias=False) # Wv for graph

        # # 학습 가능한 temperature (log scale로 양수 보장)
        self.log_scale = nn.Parameter(torch.zeros(1))
        # self.log_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, tokns_k, H_pad, mask):
        q = self.q_proj(tokns_k)          # query [bs, dim_desc_hidden]   -> [bs, r]
        k = self.k_proj(H_pad)            # key   [bs, N_max, dim_graph]  -> [bs, N_max, r]
        # # # 정규화 적용
        # q = self.q_norm(self.q_proj(tokns_k)) 
        # k = self.k_norm(self.k_proj(H_pad))


        # scale = self.scale
        scale = self.log_scale.exp().clamp(min=0.1)
        scores = torch.einsum("br,bnr->bn", q, k) / scale # [bs, N_max]


    # # --- 디버깅용 코드 시작 ---
    #     if self.training: # 학습 중에만 확인
    #         # 실제 노드가 있는 부분만 슬라이싱해서 값의 분포 확인
    #         active_scores = scores[mask] 
    #         print(f"Scores - Max: {active_scores.max().item():.4f}, "
    #             f"Min: {active_scores.min().item():.4f}, "
    #             f"Std: {active_scores.std().item():.4f}")
    #     # --- 디버깅용 코드 끝 ---



        scores = scores.masked_fill(~mask, float("-inf"))

        alpha = torch.softmax(scores, dim=-1)    # [bs, N_max]
        v = self.v_proj(H_pad)                   # [bs, N_max, dim_graph] -> [B, N_max, dim_desc_hidden]

        ctx_k = torch.einsum("bn,bnd->bd", alpha, v) # [bs, dim_desc_hidden]

        # 분자별 노드 임베딩 분산
        node_var = H_pad.std(dim=1)          # [bs, dim_graph]
        print("node embedding std mean:", node_var.mean().item())

        # attention score 분산
        print("scores std per graph:", scores[mask].std().item())
        print("H_pad[0] std across nodes:", H_pad[0, mask[0]].std(dim=0).mean())
        print("k[0] std across nodes:", k[0, mask[0]].std(dim=0).mean())
        print("scores[0]:", scores[0, mask[0]])

        # print("q:", q[0])
        # print("k:", k[0])
        # print("scores:", scores[0])
        # print("alpha:", alpha[0])

        return alpha, ctx_k # 그래프 context


class FamilyFusion(nn.Module):
    def __init__(self, dim_desc_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * dim_desc_hidden, dim_desc_hidden),
            nn.LayerNorm(dim_desc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_desc_hidden, dim_desc_hidden) # 이 부분도 굳이 필요없을거같은데
        )

    def forward(self, ctx_k, tokns_k):
        """
        c_k: [bs, dim_graph]
        tokns_k: [bs, dim_desc_hidden]

        return:
            f_k: [B, dim_desc_hidden]
        """
        fused = torch.cat([ctx_k, tokns_k, ctx_k*tokns_k, ctx_k-tokns_k], dim=-1)
        fusion_k = self.fusion_mlp(fused)

        return fusion_k
    

class FamilyAggregator(nn.Module):
    def __init__(self, dim_graph: int, dim_desc_hidden: int, dropout: float = 0.1):
        super().__init__()

        self.g_proj = nn.Linear(dim_graph, dim_desc_hidden)
        self.f_proj = nn.Linear(dim_desc_hidden, dim_desc_hidden)
        self.score = nn.Linear(dim_desc_hidden, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, family_reps, hg):
        """
        family_reps: [bs, K, dim_desc_hidden]
        hg:          [bs, dim_graph]

        return:
            beta:  [bs, K]
            h_fam: [bs, dim_desc_hidden]
        """
        g_proj = self.g_proj(hg).unsqueeze(1)           # [bs, 1, dim_desc_hidden]
        f_proj = self.f_proj(family_reps)               # [bs, K, dim_desc_hidden]

        scores = self.score(torch.tanh(f_proj + g_proj)).squeeze(-1)  # [bs, K]
        beta = torch.softmax(scores, dim=-1)                          # [bs, K]
        # print('beta....sshhhh', beta.shape)
        h_fam = torch.sum(beta.unsqueeze(-1) * family_reps, dim=1)    # [bs, dim_desc_hidden]
        h_fam = self.dropout(h_fam)

        return beta, h_fam
    

class Net(nn.Module):
    def __init__(self, dim_in: int, dim_feat_2d: int):
        super().__init__()
        self.dim_graph        = 64    # 20 → 64: graph 표현 capacity 확대
        self.dim_graph_hidden = 128
        self.dim_fc1          = 128
        self.dim_fc2          = 32
        self.dim_out          = 1
        self.drop_out         = 0.2
        self.dim_desc_hidden  = 32
        self.rank             = 16
 
        self.family_indices = family_indices
        self.family_order   = list(family_indices.keys()) # ['constitutional', 'topological', 'physicochemical', 'electronic', 'fragment']

        # ── GCN ──────────────────────────────────────────
        self.gc1 = GCNConv(dim_in, self.dim_graph_hidden)
        self.gc2 = GCNConv(self.dim_graph_hidden, self.dim_graph)
 
        # ── Descriptor pipeline ──────────────────────────
        # {fam: len(idxs)} 각 fam당 길이 dict
        family_len = {fam: len(idxs) for fam, idxs in family_indices.items()}

        # {fam: (bs, dim_desc_hiddn)}
        self.family_tokenizer = FamilyTokenizer(
            family_len=family_len,
            dim_desc_hidden=self.dim_desc_hidden,
            dropout=self.drop_out)

        # family bilinear attention
        # descriptor family로 그래프 context를 구한다
        self.family_attn = nn.ModuleDict({
            fam: FamilyBilinearAttention(
                dim_graph=self.dim_graph,
                dim_desc_hidden=self.dim_desc_hidden,
                rank=self.rank,)
            for fam in self.family_order})

        # FamilyFusion: dim_graph 인자 불필요 (ctx_proj 제거)
        self.family_fusion = nn.ModuleDict({
            fam: FamilyFusion(dim_desc_hidden=self.dim_desc_hidden)
            for fam in self.family_order})
 
        self.family_aggregator = FamilyAggregator(
            dim_graph=self.dim_graph,
            dim_desc_hidden=self.dim_desc_hidden
        )
 
        # ── Prediction head ──────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(self.dim_graph + self.dim_desc_hidden, self.dim_fc1),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.dim_fc1, self.dim_fc2),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.dim_fc2, self.dim_out),
        )
 
    def forward(self, g, desc):
        """
        Args:
            g:       DGL batched graph (node feature: g.ndata['feat'])
            feat_2d: [bs, 196]  RDKit descriptor
 
        Returns:
            out:      [bs, 1]         예측값
            attn_dict: dict[str, [B, N]]  family별 node attention weight
            beta:     [B, K]         family별 aggregation weight
        """
        # ── GCN ──────────────────────────────────────────
        h = F.relu(self.gc1(g, g.ndata['feat'])) 
        # h = F.relu(self.gc2(g, h))                # [N_total, dim_graph] (N_total: 각 배치당 총 노드 개수)
        h = self.gc2(g, h)                # [N_total, dim_graph] (N_total: 각 배치당 총 노드 개수)
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')               # [bs, dim_graph]

        # ── Descriptor tokenization ───────────────────────

        # family_inputs의 각 패밀리당 shape는 (batch size x descriptor 개수)
        # e.g.,
        # family_inputs['topological'].shape: (bs, 22)
        family_inputs = {fam: desc[:, idxs] for fam, idxs in self.family_indices.items()} # {fam: [bs, dim_desc_hidden]}

        # tokns_dict 각 패밀리당 shape는 (batch size x dim_desc_hidden)
        # e.g.,
        # tokns_dict['topological'].shape: (bs, dim_desc_hidden)
        tokns_dict = self.family_tokenizer(family_inputs)  # {fam: [bs, dim_desc_hidden]}


        # ── Bilinear attention ────────────────────────────
        # H_pad의 shape: [bs, N_max, dim_graph]
        # mask의 shape: [bs, N_max]
        H_pad, mask = split_pad_node_embeddings(g, h)

        attn_dict    = {}
        context_dict = {}

        for fam in self.family_order:
            alpha, ctx_k      = self.family_attn[fam](tokns_dict[fam], H_pad, mask)
            attn_dict[fam]    = alpha   # [bs, N_max]
            context_dict[fam] = ctx_k     # [bs, dim_desc_hidden]
        # print('attn_dict[fam]', attn_dict['constitutional'])


        # for fam in self.family_order:
        #     q = self.family_attn[fam].q_proj(tokns_dict[fam])
        #     print(f"[{fam}] q mean abs =", q.abs().mean().item(), "q std =", q.std().item())

        # for fam in self.family_order:
        #     k = self.family_attn[fam].k_proj(H_pad)
        #     print(f"[{fam}] k mean abs =", k.abs().mean().item(), "k std =", k.std().item())
        # print("q[0]:", q[0].detach().cpu())
        # print("k[0]:", k[0].detach().cpu())
        # print("scores[0]:", scores[0].detach().cpu())



        # ── Fusion & stacking ─────────────────────────────
        # [bs, k, dim_desc_hidden]
        family_reps = torch.stack([self.family_fusion[fam](context_dict[fam], tokns_dict[fam]) for fam in self.family_order], dim=1)  
        # print('family_reps', family_reps.shape)

        # ── Aggregation ───────────────────────────────────
        beta, h_fam = self.family_aggregator(family_reps, hg)  # [B,K], [B,d_desc]
 
        # ── Prediction ────────────────────────────────────
        h_final = torch.cat([hg, h_fam], dim=-1)  # [B, dim_graph + d_desc]
        out     = self.head(h_final)               # [B, 1]
 
        return out, attn_dict, beta








 
# class Net(nn.Module):
#     def __init__(self, dim_in, dim_feat_2d):
#         super(Net, self).__init__()
#         self.dim_graph = 20
#         self.dim_graph_hidden = 100
#         self.dim_fc1 = 128
#         self.dim_fc2 = 32
#         self.dim_out = 1
#         self.drop_out = 0.2
#         self.d_desc = 32
#         self.rank = 16

#         self.family_indices = family_indices

#         self.gc1 = GCNConv(dim_in, self.dim_graph_hidden)
#         self.gc2 = GCNConv(self.dim_graph_hidden, self.dim_graph)

#         family_dims = {fam: len(idxs) for fam, idxs in family_indices.items()}

#         self.family_tokenizer = FamilyTokenizer(
#             family_dims=family_dims,
#             d_desc=self.d_desc,
#             dropout=self.drop_out
#         )

#         self.family_attn = nn.ModuleDict({
#             fam: FamilyBilinearAttention(self.dim_graph, self.d_desc, rank=self.rank)
#             for fam in self.family_indices.keys()
#         })

#         self.family_fusion = nn.ModuleDict({
#             fam: FamilyFusion(self.dim_graph, self.d_desc, dropout=self.drop_out)
#             for fam in self.family_indices.keys()
#         })

#         self.family_aggregator = FamilyAggregator(
#             dim_graph=self.dim_graph,
#             d_desc=self.d_desc,
#             dropout=self.drop_out
#         )

#         self.head = nn.Sequential(
#             nn.Linear(self.dim_graph + self.d_desc, self.dim_fc1),
#             nn.ReLU(),
#             nn.Dropout(self.drop_out),
#             nn.Linear(self.dim_fc1, self.dim_fc2),
#             nn.ReLU(),
#             nn.Dropout(self.drop_out),
#             nn.Linear(self.dim_fc2, self.dim_out)
#         )

#     def forward(self, g, feat_2d):
#         h = F.relu(self.gc1(g, g.ndata['feat']))
#         h = F.relu(self.gc2(g, h))

#         g.ndata['h'] = h
#         hg = dgl.mean_nodes(g, 'h')

#         family_inputs = {
#             fam: feat_2d[:, idxs]
#             for fam, idxs in self.family_indices.items()
#         }

#         z_dict = self.family_tokenizer(family_inputs)

#         H_pad, mask = split_pad_node_embeddings(g, h)

#         context_dict = {}
#         attn_dict = {}

#         for fam, z_k in z_dict.items():
#             alpha_k, c_k = self.family_attn[fam](z_k, H_pad, mask)
#             attn_dict[fam] = alpha_k
#             context_dict[fam] = c_k

#         family_order = ["constitutional", "topological", "physicochemical", "electronic", "fragment"]
#         family_reps = []

#         for fam in family_order:
#             f_k = self.family_fusion[fam](context_dict[fam], z_dict[fam])
#             family_reps.append(f_k)

#         family_reps = torch.stack(family_reps, dim=1)

#         beta, h_fam = self.family_aggregator(family_reps, hg)

#         h_final = torch.cat([hg, h_fam], dim=-1)
#         out = self.head(h_final)

#         return out
    