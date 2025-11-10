import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GATLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=False)
        self.attn_fc = nn.Linear(2 * dim_out, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)

        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(dim_in, dim_out))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class kronecker_3(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dim_self_feat):
        super(kronecker_3, self).__init__()

        self.gc1 = MultiHeadGATLayer(dim_in, 100, num_heads)
        self.gc2 = MultiHeadGATLayer(100 * num_heads, 20, 1)

        self.fc1 = nn.Linear(20 * 3, 32)
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

        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out


class kronecker_5(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dim_self_feat):
        super(kronecker_5, self).__init__()

        self.gc1 = MultiHeadGATLayer(dim_in, 100, num_heads)
        self.gc2 = MultiHeadGATLayer(100 * num_heads, 20, 1)

        self.fc1 = nn.Linear(20 * 5, 32)
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
        
        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out


class kronecker_7(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dim_self_feat):
        super(kronecker_7, self).__init__()

        self.gc1 = MultiHeadGATLayer(dim_in, 100, num_heads)
        self.gc2 = MultiHeadGATLayer(100 * num_heads, 20, 1)

        self.fc1 = nn.Linear(20 * 7, 64)
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
        
        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class kronecker_10(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dim_self_feat):
        super(kronecker_10, self).__init__()

        self.gc1 = MultiHeadGATLayer(dim_in, 100, num_heads)
        self.gc2 = MultiHeadGATLayer(100 * num_heads, 20, 1)

        self.fc1 = nn.Linear(20 * 10, 64)
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
        
        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class kronecker_20(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dim_self_feat):
        super(kronecker_20, self).__init__()

        self.gc1 = MultiHeadGATLayer(dim_in, 100, num_heads)
        self.gc2 = MultiHeadGATLayer(100 * num_heads, 20, 1)

        self.fc1 = nn.Linear(20 * 20, 256)
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
        
        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    

class Net(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dim_self_feat):
        super(Net, self).__init__()

        self.gc1 = MultiHeadGATLayer(dim_in, 100, num_heads)
        self.gc2 = MultiHeadGATLayer(100 * num_heads, 20, 1)

        self.fc1 = nn.Linear(20 * dim_self_feat, 128)
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

        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
