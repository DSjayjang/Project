import shap
import numpy as np
import matplotlib.pyplot as plt
import torch

class SHAP:
    def __init__(self, model):
        self.model = model
        self.device = 'cuda'
        self.graph_dim = None
        self.desc_dim = None
        self.X = None
        self.shap_values = None
        self.explainer = None
        self.desc_feature_names = []

    def _extract_embeddings(self, test_data_loader):
        self.model.eval()
        z_graph_list = []
        x_desc_list = []

        with torch.no_grad():
            for bg, self_feat, target in test_data_loader:
                bg = bg.to(self.device)
                self_feat = self_feat.to(self.device) # [batch x 20]

                g_emb = self.model.get_graph_embedding(bg) # [batch x 20]
                z_graph_list.append(g_emb.cpu().numpy()) # [g_emb[0], g_emb[1], ...]               
                x_desc_list.append(self_feat.cpu().numpy()) # [self_feat[0], self_feat[1], ...]

        
        z_graph = np.concatenate(z_graph_list, axis = 0)
        x_desc = np.concatenate(x_desc_list, axis = 0)

        # # Kronecker (outer) product 수행
        # fused_list = []
        # for i in range(len(z_graph)):
        #     fused = np.outer(z_graph[i], x_desc[i]).flatten()  # shape: d_g * d_d
        #     fused_list.append(fused)
        # fused = np.stack(fused_list, axis=0)  # shape: [N, d_g * d_d]
        # self.X = fused
        # print('self.X', self.X.shape)

        # Kronecker product
        z_graph=torch.tensor(z_graph)
        x_desc=torch.tensor(x_desc)

        fused = torch.bmm(z_graph.unsqueeze(2), x_desc.unsqueeze(1))
        fused = fused.view(fused.size(0),-1)
        self.X = fused
        self.graph_dim = z_graph.shape[1] # 20
        self.desc_dim = x_desc.shape[1] # 50

        # self.X = np.concatenate([z_graph, x_desc], axis = 1)

    def _define_model_wrapper(self): # ?
        """
        SHAP Explainer용 wrapper 함수 정의
        """
        def model_wrapper(x_concat):
            x_tensor = torch.tensor(x_concat, dtype=torch.float32).to(self.device)
            return self.model.forward(x_tensor).detach().cpu().numpy()
        return model_wrapper

    def run(self, test_data_loader):
        self._extract_embeddings(test_data_loader)

        # 이건되는데 ?>> 안됨...
        # model_wrapper = self._define_model_wrapper() # ?
        # self.explainer = shap.Explainer(model_wrapper) # ?
        # self.shap_values = self.explainer(self.X, max_evals = 2*self.X.shape[1]+1)

        background = self.X[:10].numpy()
        self.X = self.X.numpy()
        model_wrapper = self._define_model_wrapper() # ?
        self.explainer = shap.KernelExplainer(model_wrapper,background) # ?
        self.shap_values = self.explainer(self.X)

        # self.explainer = shap.Explainer(model_wrapper) # ?
        # self.shap_values = self.explainer(self.X, max_evals= 2 * self.X.shape[1] + 1)
        # graph_shap = np.abs(self.shap_values.values[:, :self.graph_dim]).mean()
        # desc_shap = np.abs(self.shap_values.values[:, self.graph_dim:]).mean()
        
        # print("🔍 SHAP 분석 결과 (Test Set 기준)")
        # print(f"   🔹 GCN Embedding 평균 기여도:    {graph_shap:.4f}")
        # print(f"   🔹 Descriptor 평균 기여도:        {desc_shap:.4f}")
        # print(f"   🔹 상대 비율 (GCN / 전체):       {graph_shap / (graph_shap + desc_shap):.2%}")

        # return graph_shap, desc_shap
    
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)  # 각 feature별 평균 절댓값
        shap_matrix = mean_abs_shap.reshape(20, 50)  # (graph_dim, desc_dim)
        desc_contrib = shap_matrix.sum(axis=0)
        desc_contrib_ratio = desc_contrib / desc_contrib.sum()

        self.desc_feature_names = ['NHOHCount', 'SlogP_VSA2', 'SlogP_VSA10', 'NumAromaticRings', 'MaxEStateIndex', 
                'PEOE_VSA14', 'fr_Ar_NH', 'SMR_VSA3', 'SMR_VSA7', 'SlogP_VSA5', 
                'VSA_EState8', 'MaxAbsEStateIndex', 'PEOE_VSA2', 'fr_Nhpyrrole', 'fr_amide', 
                'SlogP_VSA3', 'BCUT2D_MRHI', 'fr_nitrile', 'MolLogP', 'PEOE_VSA10', 
                'MinPartialCharge', 'fr_Al_OH', 'fr_sulfone', 'fr_Al_COO', 'fr_nitro_arom_nonortho', 
                'fr_imidazole', 'fr_ketone_Topliss', 'PEOE_VSA7', 'fr_alkyl_halide', 'NumSaturatedHeterocycles', 
                'fr_methoxy', 'fr_phos_acid', 'fr_pyridine', 'MinAbsEStateIndex', 'fr_para_hydroxylation', 
                'fr_phos_ester', 'NumAromaticHeterocycles', 'PEOE_VSA8', 'fr_Ndealkylation2', 'PEOE_VSA5', 
                'fr_aryl_methyl', 'NumHDonors', 'fr_imide', 'fr_priamide', 'RingCount', 
                'SlogP_VSA8', 'VSA_EState4', 'SMR_VSA5', 'FpDensityMorgan3', 'FractionCSP3']
        desc_importance = list(zip(self.desc_feature_names, desc_contrib_ratio))
        desc_importance = sorted(desc_importance, key=lambda x: x[1], reverse=True)
        print("🔹 Top 10 contributing descriptors:")
        for name, val in desc_importance[:10]:
            print(f"{name:20s}  {val*100:.2f}%")


        # shap_values.values: (n_samples, 1000)
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
        shap_matrix = mean_abs_shap.reshape(20, 50)  # (graph_dim, desc_dim)

        # 그래프 임베딩별 및 descriptor별 집계
        graph_contrib = shap_matrix.sum(axis=1)  # (20,)
        desc_contrib  = shap_matrix.sum(axis=0)  # (50,)

        graph_total = graph_contrib.sum()
        desc_total  = desc_contrib.sum()
        total = graph_total + desc_total

        graph_ratio = graph_total / total
        desc_ratio  = desc_total / total

        print(f"Graph embedding contribution: {graph_ratio*100:.2f}%")
        print(f"Descriptor contribution: {desc_ratio*100:.2f}%")

    def plot_summary(self):
        # save_path = 'shapshap.png'
        # # feature_names = [f'g{i}' for i in range(self.graph_dim)] + [f'd{i}' for i in range(self.desc_dim)]
        # feature_names = [f'g{i}x {desc_feature_names[j]}' for i in range(self.graph_dim) for j in range(self.desc_dim)]
        # shap.summary_plot(self.shap_values, features=self.X, feature_names=feature_names, max_display = 20, show=False)

        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"✅ SHAP summary plot 저장 완료: {save_path}")


        save_path = 'shap_desc_summary.png'

        # --- Kronecker 구조 복원 ---
        n_samples = self.shap_values.values.shape[0]
        shap_vals = self.shap_values.values.reshape(n_samples, self.graph_dim, self.desc_dim)
        feat_vals = self.X.reshape(n_samples, self.graph_dim, self.desc_dim)

        # --- descriptor별 SHAP 및 feature 값 평균 (graph dim 방향) ---
        desc_shap_vals = shap_vals.mean(axis=1)   # (n_samples, 50)
        desc_feat_vals = feat_vals.mean(axis=1)   # (n_samples, 50)

        # --- summary plot ---
        shap.summary_plot(
            desc_shap_vals,
            features=desc_feat_vals,
            feature_names=self.desc_feature_names,
            max_display=20,
            show=False
        )

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ descriptor별 SHAP point plot 저장 완료: {save_path}")

# import shap
# import torch
# # model: multimodal fusion network (graph + descriptor)
# # x_desc: descriptor features (numpy)
# # z_graph: graph embeddings (numpy)
# X = np.concatenate([z_graph, x_desc], axis=1)
# explainer = shap.Explainer(model, X)
# shap_values = explainer(X)
# # modality split
# graph_dim = z_graph.shape[1]
# desc_dim = x_desc.shape[1]
# graph_shap = np.abs(shap_values.values[:, :graph_dim]).mean()
# desc_shap  = np.abs(shap_values.values[:, graph_dim:]).mean()

# # 참고, descriptors별 중요도
# desc_shap_vals = np.abs(shap_values.values[:, graph_dim:])  # 각 descriptor별 SHAP(샘플별)
# desc_feature_mean_importance = desc_shap_vals.mean(axis=0)  # 각 descriptor feature의 평균 SHAP 중요도

# # 예: 가장 영향력 큰 descriptor 3개 추출
# top3_idx = np.argsort(-desc_feature_mean_importance)[:3]
# top3_importance = desc_feature_mean_importance[top3_idx]
# print(top3_idx, top3_importance)