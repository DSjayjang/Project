import shap
import numpy as np
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
                g_emb = self.model.gcn(bg)
                z_graph_list.append(g_emb.cpu().numpy())
                x_desc_list.append(self_feat.cpu().numpy())

                n_collected += self_feat.shape[0]
                if n_collected >= self.max_samples:
                    break
        
        z_graph = np.concatenate(z_graph_list, axis = 0)
        x_desc = np.concatenate(x_desc_list, axis = 0)
        self.X = np.concatenate([z_graph, x_desc], axis = 1)
        self.graph_dim = z_graph.shape[1]
        self.desc_dim = x.desc.shape[1]