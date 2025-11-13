import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from configs.config import DATASET_NAME, MAX_EPOCHS, K

class SHAP:
    def __init__(self, model, device='cuda'):
        """
        model : 학습이 끝난 최종 KROVEX 모델
                - get_graph_embedding(bg)  : (B, graph_dim)
                - forward_fc(fused_flat)   : (B, 1)   # Kronecker 이후 FC 부분
        """
        self.model = model
        self.device = device

        self.graph_dim = None   # z 차원 (예: 20)
        self.desc_dim  = None   # d 차원 (예: 50)

        self.X_concat = None    # [z, d]  (N, graph_dim + desc_dim)  <-- SHAP 입력
        self.shap_values = None
        self.explainer   = None

        # descriptor 이름
        # # freesolv
        # self.desc_feature_names = [
        #     'NHOHCount', 'SlogP_VSA2', 'SlogP_VSA10', 'NumAromaticRings', 'MaxEStateIndex', 
        #     'PEOE_VSA14', 'fr_Ar_NH', 'SMR_VSA3', 'SMR_VSA7', 'SlogP_VSA5', 
        #     'VSA_EState8', 'MaxAbsEStateIndex', 'PEOE_VSA2', 'fr_Nhpyrrole', 'fr_amide', 
        #     'SlogP_VSA3', 'BCUT2D_MRHI', 'fr_nitrile', 'MolLogP', 'PEOE_VSA10', 
        #     'MinPartialCharge', 'fr_Al_OH', 'fr_sulfone', 'fr_Al_COO', 'fr_nitro_arom_nonortho', 
        #     'fr_imidazole', 'fr_ketone_Topliss', 'PEOE_VSA7', 'fr_alkyl_halide', 'NumSaturatedHeterocycles', 
        #     'fr_methoxy', 'fr_phos_acid', 'fr_pyridine', 'MinAbsEStateIndex', 'fr_para_hydroxylation', 
        #     'fr_phos_ester', 'NumAromaticHeterocycles', 'PEOE_VSA8', 'fr_Ndealkylation2', 'PEOE_VSA5', 
        #     'fr_aryl_methyl', 'NumHDonors', 'fr_imide', 'fr_priamide', 'RingCount', 
        #     'SlogP_VSA8', 'VSA_EState4', 'SMR_VSA5', 'FpDensityMorgan3', 'FractionCSP3'
        # ]
        # esol
        self.desc_feature_names = ['MolLogP', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'SMR_VSA10', 'Kappa2', 
                'BCUT2D_MWLOW', 'PEOE_VSA13', 'MinAbsPartialCharge', 'BCUT2D_CHGHI', 'PEOE_VSA6', 
                'SlogP_VSA1', 'fr_nitro', 'BalabanJ', 'SMR_VSA9', 'fr_alkyl_halide', 
                'fr_hdrzine', 'PEOE_VSA8', 'fr_Ar_NH', 'fr_imidazole', 'fr_Nhpyrrole', 
                'EState_VSA5', 'PEOE_VSA4', 'fr_ester', 'PEOE_VSA2', 'NumAromaticCarbocycles', 
                'BCUT2D_LOGPHI', 'EState_VSA11', 'fr_furan', 'EState_VSA2', 'fr_benzene', 
                'fr_sulfide', 'fr_aryl_methyl', 'SlogP_VSA10', 'HeavyAtomMolWt', 'fr_nitro_arom_nonortho', 
                'FpDensityMorgan2', 'EState_VSA8', 'fr_bicyclic', 'fr_aniline', 'fr_allylic_oxid', 
                'fr_C_S', 'SlogP_VSA7', 'SlogP_VSA4', 'fr_para_hydroxylation', 'PEOE_VSA7', 
                'fr_Al_OH_noTert', 'fr_pyridine', 'fr_phos_acid', 'fr_phos_ester', 'NumAromaticHeterocycles', 
                'EState_VSA7', 'PEOE_VSA12', 'Ipc', 'FpDensityMorgan1', 'PEOE_VSA14', 
                'fr_guanido', 'fr_benzodiazepine', 'fr_thiophene', 'fr_Ndealkylation1', 'fr_aldehyde', 
                'fr_term_acetylene', 'SMR_VSA2', 'fr_lactone']
        # # scgas
        # self.desc_feature_names =['MolMR', 'TPSA', 'fr_halogen', 'SlogP_VSA12', 'RingCount', 
        #                         'Kappa1', 'NumHAcceptors', 'NumHDonors', 'SMR_VSA7', 'SMR_VSA5',
        #                         'Chi1', 'Chi3n', 'BertzCT', 'VSA_EState8', 'NumAliphaticCarbocycles',
        #                         'HallKierAlpha', 'VSA_EState6', 'NumAromaticRings', 'Chi4n', 'PEOE_VSA7',
        #                         'SlogP_VSA5', 'VSA_EState7', 'NOCount']
    # -------------------------------------------------
    # 1) test_data_loader에서 z(graph 임베딩)와 d(descriptor) 추출
    #    그리고 X_concat = [z, d] 생성
    # -------------------------------------------------
    def _extract_embeddings(self, data_loader):
        self.model.eval()
        z_graph_list = []
        x_desc_list  = []

        with torch.no_grad():
            for bg, self_feat, target in data_loader:
                bg = bg.to(self.device)
                self_feat = self_feat.to(self.device)  # (B, desc_dim)

                # GNN → graph embedding z (B, graph_dim)
                g_emb = self.model.get_graph_embedding(bg)

                z_graph_list.append(g_emb.cpu().numpy())
                x_desc_list.append(self_feat.cpu().numpy())

        z_graph = np.concatenate(z_graph_list, axis=0)   # (N, graph_dim)
        x_desc  = np.concatenate(x_desc_list,  axis=0)   # (N, desc_dim)

        self.graph_dim = z_graph.shape[1]
        self.desc_dim  = x_desc.shape[1]

        # SHAP 입력으로 쓸 [z, d] concat
        X_concat = np.concatenate([z_graph, x_desc], axis=1)  # (N, graph_dim + desc_dim)
        self.X_concat = X_concat

        return z_graph, x_desc

    # -------------------------------------------------
    # 2) [z, d] → Kronecker → FC → y_hat 을 수행하는 wrapper
    #    (SHAP이 호출하는 모델 함수)
    # -------------------------------------------------
    def _define_model_wrapper(self):
        """
        SHAP이 보는 모델:
            입력: [z_1,...,z_g, d_1,...,d_D]  (탭형 벡터)
            내부:
                - z, d로 다시 split
                - Kronecker(z, d) = z d^T → flatten
                - model.forward_fc(fused_flat) 로 예측
        """
        graph_dim = self.graph_dim
        desc_dim  = self.desc_dim
        model     = self.model
        device    = self.device

        def model_wrapper(x_concat):
            # x_concat: (N, graph_dim + desc_dim) 또는 (graph_dim + desc_dim,) 일 수도 있음
            x_concat = np.array(x_concat)

            if x_concat.ndim == 1:
                x_concat = x_concat.reshape(1, -1)

            z_np = x_concat[:, :graph_dim]   # (N, graph_dim)
            d_np = x_concat[:, graph_dim:]   # (N, desc_dim)

            z = torch.tensor(z_np, dtype=torch.float32, device=device)  # (N, g)
            d = torch.tensor(d_np, dtype=torch.float32, device=device)  # (N, D)

            # Concatenation
            fused = torch.cat((z, d), dim = 1)

            with torch.no_grad():
                y = model.forward_fc(fused)        # (N, 1) or (N,)

            return y.detach().cpu().numpy().reshape(-1)

        return model_wrapper

    # -------------------------------------------------
    # 3) SHAP 실행
    # -------------------------------------------------
    def run(self, data_loader, background_size=50):
        """
        data_loader : 보통 test_data_loader 를 권장
        background_size : SHAP background로 사용할 샘플 수
        """
        # 1) 임베딩 & 입력 준비
        z_graph, x_desc = self._extract_embeddings(data_loader)  # self.X_concat 세팅됨
        X = self.X_concat                                       # (N, g + D)

        # 2) background 선택 (훈련/테스트 분포를 대표하는 일부 샘플)
        n_samples = X.shape[0]
        bg_size = min(background_size, n_samples)
        bg_idx = np.random.choice(n_samples, size=bg_size, replace=False)
        background = X[bg_idx]

        # 3) SHAP Explainer 생성
        model_wrapper = self._define_model_wrapper()
        self.explainer = shap.KernelExplainer(model_wrapper, background)

        # 4) SHAP 값 계산
        #    (주의: KernelExplainer는 느릴 수 있으므로, 필요하면 일부 샘플만 사용)
        self.shap_values = self.explainer(X)

        # 5) 모달리티별 기여도 계산
        self._analyze_modality_contributions()

    # -------------------------------------------------
    # 4) graph vs descriptor 기여도, descriptor별 기여도 계산
    # -------------------------------------------------
    def _analyze_modality_contributions(self):
        # shap_values.values : (N, g + D)
        vals = self.shap_values.values  # numpy array
        mean_abs = np.abs(vals).mean(axis=0)  # (g + D,)

        graph_shap = mean_abs[:self.graph_dim]        # (g,)
        desc_shap  = mean_abs[self.graph_dim:]        # (D,)

        graph_total = graph_shap.sum()
        desc_total  = desc_shap.sum()
        total = graph_total + desc_total

        graph_ratio = graph_total / total
        desc_ratio  = desc_total  / total

        print("🔍 SHAP 모달리티 기여도 (mean |SHAP| 기준)")
        print(f"   • Graph embedding contribution : {graph_ratio*100:.2f}%")
        print(f"   • Descriptor     contribution : {desc_ratio*100:.2f}%")

        # descriptor별 상대 기여도
        desc_contrib_ratio = desc_shap / desc_shap.sum()
        desc_importance = list(zip(self.desc_feature_names, desc_contrib_ratio))
        desc_importance = sorted(desc_importance, key=lambda x: x[1], reverse=True)

        print("\n🔹 Top 10 contributing descriptors (by mean |SHAP|):")
        for name, val in desc_importance[:10]:
            print(f"{name:25s}  {val*100:.2f}%")

    # -------------------------------------------------
    # 5) descriptor 기준 SHAP summary plot (beeswarm)
    # -------------------------------------------------
    def plot_summary(self, save_path=f'./results_figure/shap/shap_{DATASET_NAME}_{MAX_EPOCHS}_{K}_concat.png', max_display=10):
        """
        descriptor 축 기준으로만 SHAP beeswarm plot 출력
        - x축: descriptor 값
        - 색: SHAP value (기여도)
        """
        if self.shap_values is None or self.X_concat is None:
            raise RuntimeError("run()을 먼저 실행해서 SHAP 값을 계산해야 합니다.")

        vals = self.shap_values.values            # (N, g + D)
        X     = self.X_concat                     # (N, g + D)

        # descriptor 부분만 추출
        desc_shap_vals = vals[:, self.graph_dim:]  # (N, D)
        desc_feat_vals = X[:,  self.graph_dim:]    # (N, D)

        shap.summary_plot(
            desc_shap_vals,
            features=desc_feat_vals,
            feature_names=self.desc_feature_names,
            max_display=max_display,
            show=False
        )

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ descriptor별 SHAP point plot 저장 완료: {save_path}")
