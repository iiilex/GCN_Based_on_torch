import torch 
import torch.nn as nn
import torch.nn.functional as F


# 一个GCN层
class GCN_layer(nn.Module):
    def __init__(self, in_feature, out_feature, is_dropout, is_relu, dropout = 0.5): # 初始化
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_feature, out_feature))
        self.dropout = nn.Dropout(dropout) if is_dropout else nn.Identity()
        self.is_relu = is_relu
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, adj):
        y = adj @ x @ self.weight
        y = self.dropout(y)
        return F.relu(y) if self.is_relu else y
    
# 一个Linear层
class Linear_layer(nn.Module):
    def __init__(self, in_feature, out_feature, is_dropout, is_relu, dropout = 0.5): # 初始化
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_feature, out_feature))
        self.dropout = nn.Dropout(dropout) if is_dropout else nn.Identity()
        self.is_relu = is_relu
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, adj):
        y = x @ self.weight
        y = self.dropout(y)
        return F.relu(y) if self.is_relu else y
    
# 二层GCN
class GCN_2(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature, dropout):
        super().__init__()
        self.layer1 = GCN_layer(in_feature=in_feature , out_feature=hidden_feature, is_dropout=True,is_relu=True, dropout=dropout)
        self.layer2 = GCN_layer(in_feature=hidden_feature, out_feature=out_feature, is_dropout=False, is_relu=False)

    def forward(self, x, adj): # 前向传播
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        return x
    
    def predict(self, x, adj): # 预测
        return F.softmax(self(x,adj), dim = -1)
    
    def loss_fn(self, logits, labels, mask, weight_decay):
        # 原论文代码就是 交叉熵+第一层l2,这里保留
        loss_ce = F.cross_entropy(logits[mask], labels[mask])
        loss_l2 = weight_decay * self.layer1.weight.square().sum()
        return loss_ce + loss_l2
    
    def accuracy_fn(self, logits, labels, mask):
        preds = logits[mask].argmax(dim = -1)
        return (preds == labels[mask]).float().mean().item()
    
# 二层MLP
class MLP_2(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature, dropout):
        super().__init__()
        self.layer1 = Linear_layer(in_feature=in_feature , out_feature=hidden_feature, is_dropout=True,is_relu=True, dropout=dropout)
        self.layer2 = Linear_layer(in_feature=hidden_feature, out_feature=out_feature, is_dropout=False, is_relu=False)

    def forward(self, x, adj): # 前向传播
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        return x
    
    def predict(self, x, adj): # 预测
        return F.softmax(self(x,adj), dim = -1)
    
    def loss_fn(self, logits, labels, mask, weight_decay):
        # 原论文代码就是 交叉熵+第一层l2,这里保留
        loss_ce = F.cross_entropy(logits[mask], labels[mask])
        loss_l2 = weight_decay * self.layer1.weight.square().sum()
        return loss_ce + loss_l2
    
    def accuracy_fn(self, logits, labels, mask):
        preds = logits[mask].argmax(dim = -1)
        return (preds == labels[mask]).float().mean().item()
    
# 八层GCN
class GCN_8(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature, dropout):
        super().__init__()
        self.layer1 = GCN_layer(in_feature=in_feature , out_feature=hidden_feature, is_dropout=True,is_relu=True, dropout=dropout)
        self.layer2 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer3 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer4 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer5 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer6 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer7 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer8 = GCN_layer(in_feature=hidden_feature, out_feature=out_feature, is_dropout=False, is_relu=False)

    def forward(self, x, adj): # 前向传播
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        x = self.layer3(x, adj)
        x = self.layer4(x, adj)
        x = self.layer5(x, adj)
        x = self.layer6(x, adj)
        x = self.layer7(x, adj)
        x = self.layer8(x, adj)
        return x
    
    def predict(self, x, adj): # 预测
        return F.softmax(self(x,adj), dim = -1)
    
    def loss_fn(self, logits, labels, mask, weight_decay):
        # 原论文代码就是 交叉熵+第一层l2,这里保留
        loss_ce = F.cross_entropy(logits[mask], labels[mask])
        loss_l2 = weight_decay * self.layer1.weight.square().sum()
        return loss_ce + loss_l2
    
    def accuracy_fn(self, logits, labels, mask):
        preds = logits[mask].argmax(dim = -1)
        return (preds == labels[mask]).float().mean().item()
    
# 八层残差GCN
class ResGCN_8(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature, dropout):
        super().__init__()
        self.layer1 = GCN_layer(in_feature=in_feature , out_feature=hidden_feature, is_dropout=True,is_relu=True, dropout=dropout)
        self.layer2 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer3 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer4 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer5 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer6 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer7 = GCN_layer(in_feature=hidden_feature, out_feature=hidden_feature, is_dropout=True, is_relu=True, dropout=dropout)
        self.layer8 = GCN_layer(in_feature=hidden_feature, out_feature=out_feature, is_dropout=False, is_relu=False)
        self.proj1 = nn.Linear(in_features= in_feature, out_features= hidden_feature)
        self.proj2 = nn.Linear(in_features=hidden_feature, out_features=out_feature)

    def forward(self, x, adj): # 前向传播
        x = self.layer1(x, adj) + self.proj1(x)
        x = self.layer2(x, adj) + x
        x = self.layer3(x, adj) + x
        x = self.layer4(x, adj) + x
        x = self.layer5(x, adj) + x
        x = self.layer6(x, adj) + x
        x = self.layer7(x, adj) + x
        x = self.layer8(x, adj) + self.proj2(x)
        return x
    
    def predict(self, x, adj): # 预测
        return F.softmax(self(x,adj), dim = -1)
    
    def loss_fn(self, logits, labels, mask, weight_decay):
        # 原论文代码就是 交叉熵+第一层l2,这里保留
        loss_ce = F.cross_entropy(logits[mask], labels[mask])
        loss_l2 = weight_decay * self.layer1.weight.square().sum()
        return loss_ce + loss_l2
    
    def accuracy_fn(self, logits, labels, mask):
        preds = logits[mask].argmax(dim = -1)
        return (preds == labels[mask]).float().mean().item()
