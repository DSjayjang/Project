import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv
import dgl

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

class Net(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int = 1,
        num_layers: int = 5,
        hidden_dim: int = 64,
        final_dropout: float = 0.5,
        learn_eps: bool = True,
        graph_pooling_type: str = "sum",
        neighbor_pooling_type: str = "sum"
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.final_dropout = final_dropout
        self.graph_pooling_type = graph_pooling_type

        # GINConv layers (num_layers-1개)
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers - 1):
            if layer == 0:
                mlp = MLP(dim_in, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(GINConv(mlp, aggregator_type=neighbor_pooling_type, learn_eps=learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(dim_in, dim_out))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, dim_out))

    def _graph_pool(self, g, feat_name: str):
        if self.graph_pooling_type == "sum":
            return dgl.sum_nodes(g, feat_name)
        else:  # "average"
            return dgl.mean_nodes(g, feat_name)


    def forward(self, g):
        h = g.ndata["feat"]  # (N, dim_in)

        hidden_rep = [h]

        # message passing
        for layer in range(self.num_layers - 1):
            h = self.ginlayers[layer](g, h)   # (N, hidden_dim)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)

        # layer-wise readout + prediction sum
        out = 0.0
        for layer, h_layer in enumerate(hidden_rep):
            g.ndata["h_tmp"] = h_layer
            pooled_h = self._graph_pool(g, "h_tmp")  # (B, dim_in) or (B, hidden_dim)

            score = self.linears_prediction[layer](pooled_h)  # (B, dim_out)
            score = F.dropout(score, p=self.final_dropout, training=self.training)

            out = out + score

        return out, None