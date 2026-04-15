import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import numpy as np

# 读取数据集
dataset = Planetoid(root = "data", name = "Pubmed")
data = dataset[0]

# 取出数据，并且把要计算的部分迁移到cuda上，加速计算
device = torch.device("cuda")
x = data.x.to(device)
y = data.y.to(device)
train_mask = data.train_mask.to(device)
test_mask = data.test_mask.to(device)
num_nodes = data.num_nodes
num_features = data.num_features
num_classes = dataset.num_classes

# 增加自环
edge_index = data.edge_index.to(device)
idx = torch.arange(end = num_nodes, dtype = torch.long, device = device)
self_index = torch.stack([idx, idx], dim = 0)
print(self_index.shape)
edge_index = torch.cat([edge_index, self_index], dim = 1)
print(edge_index.shape)

# 计算 D^{-1/2}
u, v = edge_index
D = torch.bincount(u, minlength=num_nodes).float()
inv_sqrt_D = D.pow(-0.5)
inv_sqrt_D[inv_sqrt_D == float("inf")] = 0

# 在实际代码中，归一化邻接矩阵的值以边权的形式存在，来配合稀疏矩阵的形式
edge_weight = inv_sqrt_D[u] * inv_sqrt_D[v]
adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))

#定义一个GCN层
class GCN_layer(nn.Module):
    def __init__(self, in_feature, out_feature, is_dropout, is_relu): # 初始化
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_feature, out_feature))
        self.dropout = nn.Dropout(0.5) if is_dropout else nn.Identity()
        self.is_relu = is_relu
        nn.init.xavier_uniform_(self.weight)
        

    def forward(self, x, adj):
        y = adj @ x @ self.weight
        y = self.dropout(y)
        return F.relu(y) if self.is_relu else y


# 定义整个GCN模型

class GCN(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature):
        super().__init__()
        self.layer1 = GCN_layer(in_feature=in_feature , out_feature=hidden_feature, is_dropout=True,is_relu=True)
        self.layer2 = GCN_layer(in_feature=hidden_feature, out_feature=out_feature, is_dropout=False, is_relu=False)

    def forward(self, x, adj): # 前向传播
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        return x
    
    def predict(self, x, adj): # 预测
        return F.softmax(self(x,adj), dim = -1)
    
    def loss_fn(self, logits, labels, mask, weight_decay = 5e-4):
        # 原论文代码就是 交叉熵+第一层l2,这里保留
        loss_ce = F.cross_entropy(logits[mask], labels[mask])
        loss_l2 = weight_decay * self.layer1.weight.square().sum()
        return loss_ce + loss_l2
    
    def accuracy_fn(self, logits, labels, mask):
        preds = logits[mask].argmax(dim = -1)
        return (preds == labels[mask]).float().mean().item()

# 开始训练 
model = GCN(in_feature=num_features, hidden_feature=16, out_feature=num_classes)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
epochs = 150
model.train()

acc_history = []
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(x, adj)
    loss = model.loss_fn(logits, y, train_mask)
    loss.backward()
    optimizer.step()

    model.eval()
    acc = model.accuracy_fn(logits, y, test_mask)
    acc_history.append(acc)
    loss_history.append(loss.item())

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

model.eval()
with torch.no_grad():
    logits = model(x, adj)
    test_acc = model.accuracy_fn(logits, y, test_mask)
    print(f"Final_Acc = {test_acc:.4f}")

idx = [i for i in range(1,epochs+1)]

fig, ax1 = plt.subplots(figsize=(14, 5))

# ========== 左侧 Y 轴：Loss ==========
color_loss = '#FF5733'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color=color_loss, fontsize=12, fontweight='bold')
loss_line1 = ax1.plot(idx, loss_history, color=color_loss, 
                       linewidth=2, alpha=0.8, label='loss')
ax1.grid(True, axis='y', alpha=0.3)

# ========== 右侧 Y 轴：Accuracy ==========
ax2 = ax1.twinx()
color_acc = '#3A9BDC'
ax2.set_ylabel('Accuracy', color=color_acc, fontsize=12, fontweight='bold')
acc_line2 = ax2.plot(idx, acc_history, color=color_acc, 
                     linestyle='--', linewidth=2, alpha=0.8, label='acc')

# ========== 合并图例 ==========
lines_1 = loss_line1
lines_2 = acc_line2
all_lines = lines_1 + lines_2
labels = [l.get_label() for l in all_lines]
plt.legend(all_lines, labels, loc='upper left', framealpha=0.9)

# 统一标题
plt.tight_layout()
plt.savefig('training_curves_dual_axis.png', dpi=300, bbox_inches='tight')
plt.show()