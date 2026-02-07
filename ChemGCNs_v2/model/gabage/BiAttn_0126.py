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


class Net(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc):
        super(Net, self).__init__()

        self.dim_in = dim_in
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        d_g = 20
        d_h = 64
        drop = 0.3

        mlp_hidden1 = 128
        mlp_hidden2 = 32
        dim_out = 1

        # GCN layer
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, d_g)

        # Bilinear attention blocks
        self.proj_g = nn.Linear(d_g, d_h)
        self.proj_2d = nn.Linear(dim_2d_desc, d_h)
        self.proj_3d = nn.Linear(dim_3d_desc, d_h)

        # bilinear weight W: (d_h, d_h)
        self.W = nn.Parameter(torch.empty(d_g, dim_2d_desc))
        nn.init.xavier_uniform_(self.W)

        self.W2 = nn.Parameter(torch.empty(d_h, d_h))
        self.W3 = nn.Parameter(torch.empty(d_h, d_h))
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)

        # MLP head
        # self.fc1 = nn.Linear(d_g + dim_2d_desc, mlp_hidden1)
        self.fc1 = nn.Linear((d_g+1) * (dim_2d_desc+1), mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3 = nn.Linear(mlp_hidden2, dim_out)

        self.bn1 = nn.BatchNorm1d(mlp_hidden1)
        self.bn2 = nn.BatchNorm1d(mlp_hidden2)

        self.dropout = nn.Dropout(drop)

    def forward(self, g, desc_2d, desc_3d):
        """
        g:       (B, d_g)          graph embedding
        desc_2d: (B, dim_2d_desc)  2d descriptors
        desc_3d: (B, dim_3d_desc)  3d descriptors
        """
        # GCN
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')   # (B, 20)

        scores0 = hg @ self.W
        scores = scores0 * desc_2d
        attn = torch.softmax(scores, dim = -1)
        desc_att = attn * desc_2d
        print('attn.shape', attn.shape)
        print('attn', attn)
        
        # tensor fusion
        batch_size = g.batch_size
        ones = torch.ones(batch_size, 1, device=hg.device, dtype=hg.dtype)

        hg = torch.cat((hg, ones), dim = 1)
        desc_att = torch.cat((desc_att, ones), dim = 1)

        fusion = torch.bmm(hg.unsqueeze(2), desc_att.unsqueeze(1)).view(batch_size, -1)
        # MLP
        out = F.relu(self.bn1(self.fc1(fusion)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)                # (B, dim_out)

        return out


class Net2(nn.Module):
    def __init__(self, dim_in, dim_2d_desc, dim_3d_desc):
        super(Net2, self).__init__()

        self.dim_in = dim_in
        self.dim_2d_desc = dim_2d_desc
        self.dim_3d_desc = dim_3d_desc
        d_g = 20
        d_h = 64
        drop = 0.3

        mlp_hidden1 = 128
        mlp_hidden2 = 32
        dim_out = 1

        # GCN layer
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, d_g)

        # Bilinear attention blocks
        self.proj_g = nn.Linear(d_g, d_h)
        self.proj_2d = nn.Linear(dim_2d_desc, d_h)
        self.proj_3d = nn.Linear(dim_3d_desc, d_h)

        # bilinear weight W: (d_h, d_h)
        self.W = nn.Parameter(torch.empty(d_g, dim_2d_desc))
        nn.init.xavier_uniform_(self.W)

        self.W2 = nn.Parameter(torch.empty(d_h, d_h))
        self.W3 = nn.Parameter(torch.empty(d_h, d_h))
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)

        # MLP head
        # self.fc1 = nn.Linear(d_g + dim_2d_desc, mlp_hidden1)
        self.fc1 = nn.Linear((d_g+1) * (dim_2d_desc+1), mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3 = nn.Linear(mlp_hidden2, dim_out)

        self.bn1 = nn.BatchNorm1d(mlp_hidden1)
        self.bn2 = nn.BatchNorm1d(mlp_hidden2)

        self.dropout = nn.Dropout(drop)

    def forward(self, g, desc_2d, desc_3d):
        """
        g:       (B, d_g)          graph embedding
        desc_2d: (B, dim_2d_desc)  2d descriptors
        desc_3d: (B, dim_3d_desc)  3d descriptors
        """
        # GCN
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')   # (B, 20)

        scores = hg @ self.W
        attn = torch.softmax(scores, dim = -1)
        desc_att = attn * desc_2d

        # tensor fusion
        batch_size = g.batch_size
        ones = torch.ones(batch_size, 1, device=hg.device, dtype=hg.dtype)

        hg = torch.cat((hg, ones), dim = 1)
        desc_att = torch.cat((desc_att, ones), dim = 1)

        fusion = torch.bmm(hg.unsqueeze(2), desc_att.unsqueeze(1)).view(batch_size, -1)
        # MLP
        out = F.relu(self.bn1(self.fc1(fusion)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)                # (B, dim_out)

        return out
