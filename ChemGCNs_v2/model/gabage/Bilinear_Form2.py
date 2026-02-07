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

        self.W = nn.Linear(dim_self_feat, 20, bias=True)

        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)

        self.gamma = nn.Linear(dim_self_feat, 20)
        self.beta  = nn.Linear(dim_self_feat, 20)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # # bilinear form
        # A = self.W(self_feat)# (B ,20)
        # hg = A * hg # (B, 20)
        
        gamma = torch.sigmoid(self.gamma(self_feat))  # (0,1) or use softplus
        beta  = self.beta(self_feat)
        hg = hg * gamma + beta

        # fully connected networks
        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)

        return out
    
class EnhancedFiLMNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat): # dim_self_feat = 2D + 3D 합친 차원
        super(EnhancedFiLMNet, self).__init__()

        # 1. Graph Encoder: 차원을 좀 더 넉넉하게 유지 (20 -> 64)
        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20) 

        # 2. FiLM Generator: 2-layer MLP로 표현력 강화
        self.gamma_net = nn.Sequential(
            nn.Linear(dim_self_feat, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.Sigmoid() 
        )
        self.beta_net = nn.Sequential(
            nn.Linear(dim_self_feat, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )

        # 3. Post-Fusion MLP: FiLM 이후 정보를 처리
        # 입력이 64이므로 fc1을 64에서 시작하거나 확장
        self.fc1 = nn.Linear(20, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, dim_out)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        # Graph Embedding
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = self.gc2(g, h) # 마지막 GC layer는 활성화 전 상태로 뽑기
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h') # (B, 64)

        # FiLM 파라미터 생성
        gamma = self.gamma_net(self_feat)
        beta = self.beta_net(self_feat)

        # FiLM 변조 (Modulation)
        # hg_mod = (1 + gamma) * hg + beta  # Residual 방식 (학습에 더 유리함)
        hg_mod = hg * gamma + beta

        # MLP 단계
        out = F.relu(self.bn1(self.fc1(hg_mod)))
        out = self.dropout(out)
        
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out