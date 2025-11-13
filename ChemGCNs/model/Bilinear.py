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
    

class bilinear_Net_3(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(bilinear_Net_3, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.W = nn.Parameter(torch.randn(20, dim_self_feat))

        self.fc1 = nn.Linear(dim_self_feat, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, dim_out)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # bilinear form
        A = hg @ self.W
        hg = A * self_feat

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    
    
class bilinear_Net_5(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(bilinear_Net_5, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.W = nn.Parameter(torch.randn(20, dim_self_feat))

        self.fc1 = nn.Linear(dim_self_feat, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, dim_out)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # bilinear form
        A = hg @ self.W
        hg = A * self_feat

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class bilinear_Net_7(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(bilinear_Net_7, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.W = nn.Parameter(torch.randn(20, dim_self_feat))

        self.fc1 = nn.Linear(dim_self_feat, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, dim_out)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # bilinear form
        A = hg @ self.W
        hg = A * self_feat

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class bilinear_Net_10(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(bilinear_Net_10, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.W = nn.Parameter(torch.randn(20, dim_self_feat))

        self.fc1 = nn.Linear(dim_self_feat, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, dim_out)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # bilinear form
        A = hg @ self.W
        hg = A * self_feat

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class bilinear_Net_20(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(bilinear_Net_20, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.W = nn.Parameter(torch.randn(20, dim_self_feat))

        self.fc1 = nn.Linear(dim_self_feat, 256) 
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # bilinear form
        A = hg @ self.W
        hg = A * self_feat

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out


# Bilinear Form
class bilinear_Net(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(bilinear_Net, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.W = nn.Parameter(torch.randn(20, dim_self_feat))

        self.fc1 = nn.Linear(dim_self_feat, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # bilinear form
        A = hg @ self.W
        hg = A * self_feat

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out