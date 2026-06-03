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
        elif self.merge == 'mean':
            return torch.stack(head_outs, dim=0).mean(dim=0)


class Net(nn.Module):
    def __init__(self, dim_in, num_heads=4):
        super(Net, self).__init__()
        dim_out = 1

        self.gc1 = MultiHeadGATLayer(dim_in, 100, num_heads, merge='cat')
        self.gc2 = MultiHeadGATLayer(100 * num_heads, 20, num_heads, merge='mean')

        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        out = F.relu(self.fc1(hg))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out, None
