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

# TFN
class Net(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat, dim_3d_feat):
        super(Net, self).__init__()

        self.dim_graph_emb = 20
        self.dim_self_feat = dim_self_feat
        self.dim_3d_feat = dim_3d_feat

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, self.dim_graph_emb)

        self.fc1 = nn.Linear((self.dim_graph_emb+1) * (dim_self_feat+1) * (dim_3d_feat+1), 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(8)

        self.dropout = nn.Dropout(0.15)

    def forward(self, g, self_feat, x3d):
        batch_size = g.batch_size

        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # tensor fusion
        ones = torch.ones(batch_size, 1, device=hg.device, dtype=hg.dtype)

        hg = torch.cat((hg, ones), dim = 1)
        self_feat = torch.cat((self_feat, ones), dim = 1)
        x3d = torch.cat((x3d, ones), dim = 1)

        fusion_tensor = torch.bmm(hg.unsqueeze(2), self_feat.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(-1, (self.dim_graph_emb+1) * (self.dim_self_feat+1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, x3d.unsqueeze(1)).view(batch_size, -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(fusion_tensor)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out