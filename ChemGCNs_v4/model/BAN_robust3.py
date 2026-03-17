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

# class DualPathEncoder(nn.Module):
#     """
#     GCN 임베딩 hg를 두 직교 성분으로 분리.

#     h_scaf : scaffold topology 성분  → ScaffoldAuxHead가 supervise
#     h_phys : physicochemical 성분    → BilinearAttention의 query로 사용

#     추가 파라미터:
#       2 × (dim_g² )  [proj_scaf, proj_phys]
#       2 × (dim_g  )  [LayerNorm affine]
#       dim_g = 20 → 총 880개 파라미터 (극소)
#     """
#     def __init__(self, dim_graph: int):
#         super().__init__()
#         self.proj_scaf = nn.Linear(dim_graph, dim_graph, bias=False)
#         self.proj_phys = nn.Linear(dim_graph, dim_graph, bias=False)
#         self.norm_scaf = nn.LayerNorm(dim_graph)
#         self.norm_phys = nn.LayerNorm(dim_graph)

#         # 초기화: 두 projection이 처음부터 서로 다른 방향을 바라보도록
#         nn.init.orthogonal_(self.proj_scaf.weight)
#         nn.init.orthogonal_(self.proj_phys.weight)

#     def forward(self, hg: torch.Tensor):
#         """
#         Args:
#             hg : (bs, dim_graph)
#         Returns:
#             h_scaf : (bs, dim_graph)  scaffold 성분
#             h_phys : (bs, dim_graph)  physicochemical 성분
#         """
#         # h_scaf = self.norm_scaf(F.relu(self.proj_scaf(hg)))
#         # h_phys = self.norm_phys(F.relu(self.proj_phys(hg)))
#         h_scaf = self.norm_scaf(self.proj_scaf(hg))
#         h_phys = self.norm_phys(self.proj_phys(hg))
#         return h_scaf, h_phys

#     @staticmethod
#     def orth_loss(h_scaf: torch.Tensor, h_phys: torch.Tensor) -> torch.Tensor:
#         """
#         샘플 단위 직교 정규화 손실.

#         L_orth = E[cos²(h_scaf_i, h_phys_i)]  → 0

#         배치 수준 공분산 대신 샘플 수준 코사인 유사도를 사용:
#           - 배치 크기에 덜 민감
#           - 각 분자 표현 내부에서 두 성분이 직교임을 직접 강제
#         """
#         h_s = F.normalize(h_scaf, dim=-1)   # (bs, dim_g)
#         h_p = F.normalize(h_phys, dim=-1)   # (bs, dim_g)
#         cos_sim = (h_s * h_p).sum(dim=-1)   # (bs,)
#         return cos_sim.pow(2).mean()
    
class DualPathEncoder(nn.Module):
    def __init__(self, dim_graph: int, negative_slope: float = 0.1):
        super().__init__()
        self.proj_scaf = nn.Linear(dim_graph, dim_graph, bias=False)
        self.proj_phys = nn.Linear(dim_graph, dim_graph, bias=False)

        self.norm_scaf = nn.LayerNorm(dim_graph)
        self.norm_phys = nn.LayerNorm(dim_graph)

        self.act = nn.LeakyReLU(negative_slope=negative_slope)

        nn.init.orthogonal_(self.proj_scaf.weight)
        nn.init.orthogonal_(self.proj_phys.weight)

    def forward(self, hg: torch.Tensor):
        # residual branch
        scaf_delta = self.act(self.proj_scaf(hg))
        phys_delta = self.act(self.proj_phys(hg))

        # residual connection + norm
        h_scaf = self.norm_scaf(hg + scaf_delta)
        h_phys = self.norm_phys(hg + phys_delta)

        return h_scaf, h_phys

    @staticmethod
    def orth_loss(h_scaf: torch.Tensor, h_phys: torch.Tensor) -> torch.Tensor:
        h_s = F.normalize(h_scaf, dim=-1)
        h_p = F.normalize(h_phys, dim=-1)
        cos_sim = (h_s * h_p).sum(dim=-1)
        return cos_sim.pow(2).mean()

class ScaffoldAuxHead(nn.Module):
    """
    h_scaf가 실제로 scaffold 정체성을 인코딩하도록
    보조 분류 손실로 supervise.

    이 head가 없으면 두 projection이 단순히 임의의 직교 부분공간으로
    수렴할 뿐, scaffold vs physicochemical이라는 의미론적 분리가 보장되지 않음.

    추가 파라미터:
      dim_graph × n_scaffolds
      (dim_g=20, n_scaffolds≈300 → ~6,000개, 여전히 parameter-efficient)
    """
    def __init__(self, dim_graph: int, n_scaffolds: int):
        super().__init__()
        self.clf = nn.Linear(dim_graph, n_scaffolds)

    def forward(self, h_scaf: torch.Tensor) -> torch.Tensor:
        return self.clf(h_scaf)  # (bs, n_scaffolds)

    def loss(self, h_scaf: torch.Tensor, scaffold_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(h_scaf)
        return F.cross_entropy(logits, scaffold_ids)


def scaffold_attention_consistency_loss(
    attn_list: list,
    scaffold_ids: torch.Tensor,
) -> torch.Tensor:
    """
    동일 scaffold 내 분자들의 attention 분포를 일관되게 유도하는 손실.

    [화학적 의미]
    같은 scaffold를 공유하는 분자들은 구조적 골격이 동일하다.
    따라서 h_phys(physicochemical query)가 제대로 분리되었다면,
    같은 scaffold 내에서 bilinear attention은 동일한 descriptor 군
    (예: LogP, TPSA 계열)에 주목해야 한다.
    이 일관성을 KL divergence로 강제함으로써:
      - attention이 scaffold topology가 아닌 physicochemical 신호에 anchoring되도록 학습
      - 결과적으로 unseen scaffold에서도 안정적인 descriptor 활성화 패턴 유지

    [Bilinear attention과의 연결]
    BAN의 glimpse-wise iterative refinement에서 마지막 glimpse의
    attention이 가장 정제된 physicochemical 정보를 반영하므로
    해당 attn을 consistency 대상으로 사용.

    추가 파라미터: 없음 (순수 loss function)

    Args:
        attn_list : BilinearAttention.forward()의 반환값, len=glimpse
                    각 원소: (bs, M)
        scaffold_ids : (bs,) LongTensor, 같은 Murcko scaffold → 같은 정수

    Returns:
        L_cons : scalar tensor
    """
    # 가장 정제된 마지막 glimpse attention 사용
    attn = attn_list[-1]                      # (bs, M)
    device = attn.device

    total_kl = torch.zeros(1, device=device)
    count = 0

    for s_id in scaffold_ids.unique():
        mask = (scaffold_ids == s_id)
        if mask.sum() < 2:
            continue

        group_attn = attn[mask]               # (n_s, M)

        # 그룹 평균 분포를 anchor로 (detach: anchor를 고정해 안정적으로 수렴)
        mean_attn = group_attn.mean(0, keepdim=True).detach()  # (1, M)

        # KL(각 분자 || 그룹 평균): 개별 분포를 평균으로 끌어당김
        kl = F.kl_div(
            F.log_softmax(group_attn, dim=-1),
            F.softmax(mean_attn.expand_as(group_attn), dim=-1),
            reduction='batchmean',
        )
        total_kl = total_kl + kl
        count += 1

    return total_kl / max(count, 1)


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

class Net(nn.Module):
    def __init__(self, dim_in: int, dim_desc_2d: int, n_scaffolds: int = None):
        super().__init__()
        self.dim_in      = dim_in
        self.dim_desc_2d = dim_desc_2d
        self.use_scaffold = n_scaffolds is not None
        
        # ── 하이퍼파라미터 ─────────────────────────────────────────────────────
        self.dim_graph = 20
        self.dim_fc1   = 256
        self.dim_fc2   = 64
        self.dim_out   = 1
        self.drop_out  = 0.2
        self.d_t       = 64
        self.K         = 16
        self.glimpse   = 2

        # ── GCN (변경 없음) ───────────────────────────────────────────────────
        self.gc1 = GCNConv(dim_in, 100)
        self.gc2 = GCNConv(100, self.dim_graph)

        # ── Prediction Head (변경 없음) ───────────────────────────────────────
        self.fc1     = nn.Linear(self.dim_graph, self.dim_fc1)
        self.fc2     = nn.Linear(self.dim_fc1,   self.dim_fc2)
        self.fc3     = nn.Linear(self.dim_fc2,   self.dim_out)
        self.bn1     = nn.LayerNorm(self.dim_fc1)
        self.bn2     = nn.LayerNorm(self.dim_fc2)
        self.dropout = nn.Dropout(self.drop_out)

        # ── Descriptor Tokenizer (변경 없음) ──────────────────────────────────
        self.tokenizer = Descriptor_Tokenizer(self.dim_desc_2d, self.d_t)

        # ── Bilinear Attention: query를 h_phys로 교체 (구조 변경 없음) ─────────
        self.ban = BilinearAttention(
            d_q=self.dim_graph, d_t=self.d_t,
            glimpse=self.glimpse, K=self.K,
        )
        if self.use_scaffold:
            # ── [NEW] Dual-Path Encoder ───────────────────────────────────────────
            self.dual_encoder = DualPathEncoder(self.dim_graph)
            # ── [NEW] Scaffold Auxiliary Head (n_scaffolds 제공 시 활성화) ─────────
            self.scaffold_aux = ScaffoldAuxHead(self.dim_graph, n_scaffolds)
        else:
            self.dual_encoder = None
            self.scaffold_aux = None

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, g, desc_2d):
        # 1) GCN forward
        h  = F.relu(self.gc1(g, g.ndata['feat']))
        h  = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')              # (bs, dim_graph)

        # 3) Descriptor tokenization
        desc_tks = self.tokenizer(desc_2d)       # (bs, dim_desc_2d, d_t)

        if self.use_scaffold:
            # 2) [NEW] Dual-Path 분리
            #    h_scaf : scaffold topology 성분 → L_aux, L_orth에만 사용
            #    h_phys : physicochemical 성분   → BAN query로 사용
            h_scaf, h_phys = self.dual_encoder(hg)
            # # 4) Bilinear Attention — query: h_phys (기존 hg 대신)
            fused, attn_list = self.ban(h_phys, desc_tks)
        else:
            h_scaf, h_phys = None, None
            fused, attn_list = self.ban(hg, desc_tks)

        # 5) Prediction
        out = F.relu(self.bn1(self.fc1(fused)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out, attn_list, h_scaf, h_phys

    # ─────────────────────────────────────────────────────────────────────────
    # def compute_scaffold_losses(
    #     self,
    #     h_scaf       : torch.Tensor,
    #     h_phys       : torch.Tensor,
    #     attn_list    : list,
    #     scaffold_ids : torch.Tensor,
    #     lambda_orth  : float = 0.10,
    #     lambda_aux   : float = 0.50,
    #     lambda_cons  : float = 0.05
    # ) -> dict:
    #     """
    #     세 가지 scaffold-robustness 손실을 계산 후 가중합 반환.

    #     L_orth : h_scaf ⊥ h_phys  강제 (항상 적용)
    #     L_aux  : h_scaf → scaffold 분류 (ScaffoldAuxHead 있을 때)
    #     L_cons : 동일 scaffold 내 attention 일관성 (scaffold_ids 있을 때)
    #     """
    #     losses = {}

    #     # ① 직교 손실 (파라미터 없음, 항상 계산)
    #     losses['orth'] = DualPathEncoder.orth_loss(h_scaf, h_phys)

    #     # ② 보조 scaffold 분류 손실
    #     if self.scaffold_aux is not None and scaffold_ids is not None:
    #         losses['aux'] = self.scaffold_aux.loss(h_scaf, scaffold_ids)
    #     else:
    #         losses['aux'] = torch.zeros(1, device=h_scaf.device).squeeze()

    #     # ③ Attention consistency 손실
    #     if scaffold_ids is not None:
    #         losses['cons'] = scaffold_attention_consistency_loss(
    #             attn_list, scaffold_ids
    #         )
    #     else:
    #         losses['cons'] = torch.zeros(1, device=h_scaf.device).squeeze()

    #     losses['total'] = (
    #         lambda_orth * losses['orth'] +
    #         lambda_aux  * losses['aux']  +
    #         lambda_cons * losses['cons']
    #     )
    #     return losses
    def compute_scaffold_losses(
        self,
        h_scaf: torch.Tensor,
        h_phys: torch.Tensor,
        attn_list: list,
        scaffold_ids: torch.Tensor,
        lambda_orth: float = 0.10,
        lambda_aux: float = 0.50,
        lambda_cons: float = 0.05,
    ) -> dict:

        device = attn_list[-1].device if attn_list is not None else h_scaf.device
        zero = torch.tensor(0.0, device=device)

        losses = {
            'orth': zero,
            'aux': zero,
            'cons': zero,
            'total': zero,
        }

        if not self.use_scaffold:
            return losses

        losses['orth'] = DualPathEncoder.orth_loss(h_scaf, h_phys)

        if self.scaffold_aux is not None and scaffold_ids is not None:
            losses['aux'] = self.scaffold_aux.loss(h_scaf, scaffold_ids)

        if scaffold_ids is not None:
            losses['cons'] = scaffold_attention_consistency_loss(attn_list, scaffold_ids)

        losses['total'] = (
            lambda_orth * losses['orth']
            + lambda_aux * losses['aux']
            + lambda_cons * losses['cons']
        )
        return losses
# 장점 compared to KROVEX
# parameter efficient
# interpretability to descriptors with attention mechanism
# good when scffold split