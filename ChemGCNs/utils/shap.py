import shap
import numpy as np
import matplotlib.pyplot as plt
import torch

device = 'cuda'

class SHAP:
    def __init__(self, model, max_samples = 10):
        self.model = model
        self.device = 'cuda'
        self.max_samples = max_samples
        self.graph_dim = None
        self.desc_dim = None
        self.X = None
        self.shap_values = None
        self.explainer = None

    def _extract_embeddings(self, test_data_loader):
        self.model.eval()
        z_graph_list = []
        x_desc_list = []
        n_collected = 0 # 왜필요?

        with torch.no_grad():
            for bg, self_feat, target in test_data_loader:
                bg = bg.to(self.device)
                self_feat = self_feat.to(self.device)

                # 수정 필요
                g_emb = self.model.get_graph_embedding(bg)
                z_graph_list.append(g_emb.cpu().numpy())
                x_desc_list.append(self_feat.cpu().numpy())

                n_collected += self_feat.shape[0]
                # if n_collected >= self.max_samples:
                #     break
        
        z_graph = np.concatenate(z_graph_list, axis = 0)
        x_desc = np.concatenate(x_desc_list, axis = 0)

        # ✅ Kronecker (outer) product 수행
        fused_list = []
        for i in range(len(z_graph)):
            fused = np.outer(z_graph[i], x_desc[i]).flatten()  # shape: d_g * d_d
            fused_list.append(fused)
        fused = np.stack(fused_list, axis=0)  # shape: [N, d_g * d_d]
        self.X = fused

        # self.X = np.concatenate([z_graph, x_desc], axis = 1)
        self.graph_dim = z_graph.shape[1]
        self.desc_dim = x_desc.shape[1]

    def _define_model_wrapper(self): # ?
        """
        SHAP Explainer용 wrapper 함수 정의
        """
        def model_wrapper(x_concat):
            x_tensor = torch.tensor(x_concat, dtype=torch.float32).to(self.device)
            return self.model.head(x_tensor).detach().cpu().numpy()
        return model_wrapper
    
    def run(self, test_data_loader):
        self._extract_embeddings(test_data_loader)

        def model_wrapper(x_concat):
            x_tensor = torch.tensor(x_concat, dtype=torch.float32).to(self.device)
            return self.model.forward_fc(x_tensor).detach().cpu().numpy()

        # model_wrapper = self._define_model_wrapper() # ?
        self.explainer = shap.Explainer(model_wrapper, self.X) # ?
        self.shap_values = self.explainer(self.X, max_evals = 2*self.X.shape[1]+1)

        graph_shap = np.abs(self.shap_values.values[:, :self.graph_dim]).mean()
        desc_shap = np.abs(self.shap_values.values[:, self.graph_dim:]).mean()
        
        print("🔍 SHAP 분석 결과 (Test Set 기준)")
        print(f"   🔹 GCN Embedding 평균 기여도:    {graph_shap:.4f}")
        print(f"   🔹 Descriptor 평균 기여도:        {desc_shap:.4f}")
        print(f"   🔹 상대 비율 (GCN / 전체):       {graph_shap / (graph_shap + desc_shap):.2%}")

        return graph_shap, desc_shap
    
    def plot_summary(self):
        save_path = 'shapshap.png'
        # feature_names = [f'g{i}' for i in range(self.graph_dim)] + [f'd{i}' for i in range(self.desc_dim)]
        feature_names = [f'g{i}x d{j}' for i in range(self.graph_dim) for j in range(self.desc_dim)]
        shap.summary_plot(self.shap_values, features=self.X, feature_names=feature_names, max_display = 20, show=False)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ SHAP summary plot 저장 완료: {save_path}")

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

