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
    
class concat_Net_3(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(concat_Net_3, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.fc1 = nn.Linear(20 + dim_self_feat, 10)
        self.fc2 = nn.Linear(10, dim_out)


    def forward(self, g, self_feat):

        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        hg = torch.cat((hg, self_feat), dim=1)
        
        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out
    
class concat_Net_5(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(concat_Net_5, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.fc1 = nn.Linear(20 + dim_self_feat, 10)
        self.fc2 = nn.Linear(10, dim_out)


    def forward(self, g, self_feat):

        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        hg = torch.cat((hg, self_feat), dim=1)
        
        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out

class concat_Net_7(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(concat_Net_7, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.fc1 = nn.Linear(20 + dim_self_feat, 10)
        self.fc2 = nn.Linear(10, dim_out)


    def forward(self, g, self_feat):

        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        hg = torch.cat((hg, self_feat), dim=1)
        
        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out
    
class concat_Net_10(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(concat_Net_10, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.fc1 = nn.Linear(20 + 10, 10)
        self.fc2 = nn.Linear(10, dim_out)


    def forward(self, g, self_feat):

        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        hg = torch.cat((hg, self_feat), dim=1)
        
        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out
    
class concat_Net_20(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(concat_Net_20, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.fc1 = nn.Linear(20 + 20, 10)
        self.fc2 = nn.Linear(10, dim_out)


    def forward(self, g, self_feat):

        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        hg = torch.cat((hg, self_feat), dim=1)
        
        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out
    
class concat_Net(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(concat_Net, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.fc1 = nn.Linear(20 + dim_self_feat, 10)
        self.fc2 = nn.Linear(10, dim_out)

    # def forward(self, g, self_feat):

    #     h = F.relu(self.gc1(g, g.ndata['feat']))
    #     h = F.relu(self.gc2(g, h))
    #     g.ndata['h'] = h

    #     hg = dgl.mean_nodes(g, 'h')
    #     hg = torch.cat((hg, self_feat), dim=1)
        
    #     out = F.relu(self.fc1(hg))
    #     out = self.fc2(out)

    #     return out
    

    def get_graph_embedding(self, g):
        """2-layer GCN 후 그래프 임베딩 반환"""
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')   # [batch, 20]
        return hg

    def get_fused_representation(self, g, self_feat):
        """GCN + Concatenation fusion 후 FC layer 입력 전 벡터 반환"""
        hg = self.get_graph_embedding(g)
        hg = torch.cat((hg, self_feat), dim=1)
        return hg

    def forward_fc(self, hg):
        """FC layer 부분만 따로 실행"""
        out = F.relu(self.fc1(hg))
        out = self.fc2(out)
        return out

    def forward(self, g, self_feat):
        fused = self.get_fused_representation(g, self_feat)
        return self.forward_fc(fused)