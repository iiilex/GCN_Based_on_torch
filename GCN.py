import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import copy
import argparse

### -------------------------------------------------------- ###

#定义所有需要的参数
parser = argparse.ArgumentParser(description="超参数")
parser.add_argument("--hidden_features", type=int, default=16, help="隐藏层特征数")
parser.add_argument("--lr", type=float, default=0.01, help="学习率")
parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout率")
parser.add_argument("--wd", type=float, default=5e-4, help="权重衰减")
parser.add_argument("--patience", type=int, default=30, help="早停轮数")
parser.add_argument("--dataset_type", type=int, default=0, help="0为Cora, 1为Citeseer, 2为Pubmed, 默认Cora")

### -------------------------------------------------------- ###

# 计算权重矩阵
def get_adj(edge_index, num_nodes): 
    # 计算 D^{-1/2}
    u, v = edge_index
    D = torch.bincount(u, minlength=num_nodes).float()
    inv_sqrt_D = D.pow(-0.5)
    inv_sqrt_D[inv_sqrt_D == float("inf")] = 0

    # 在实际代码中，归一化邻接矩阵的值以边权的形式存在，来配合稀疏矩阵的形式
    edge_weight = inv_sqrt_D[u] * inv_sqrt_D[v]
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    return adj

### -------------------------------------------------------- ###

#定义一个GCN层
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

### -------------------------------------------------------- ###

# 定义整个GCN模型
class GCN(nn.Module):
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

### -------------------------------------------------------- ###

def train_for_one_epoch(model, optimizer, x, adj, y, train_mask, weight_decay): # 一次训练
    model.train()
    optimizer.zero_grad()
    logits = model(x, adj)
    loss = model.loss_fn(logits, y, train_mask, weight_decay)
    loss.backward()
    optimizer.step()
    return logits, loss

### -------------------------------------------------------- ###

def eval_for_one_epoch(model, logits, y, val_mask, weight_decay): # 一次评估
    model.eval()
    with torch.no_grad():
        acc = model.accuracy_fn(logits, y, val_mask)
        val_loss = model.loss_fn(logits, y, val_mask, weight_decay)
    return acc, val_loss

### -------------------------------------------------------- ###

# 用最好的模型结果来计算测试集
def final_test(model, x, adj, y, test_mask, best_epoch):
    model.eval()
    with torch.no_grad():
        logits = model(x, adj)
        test_acc = model.accuracy_fn(logits, y, test_mask)
        print(f"Best_epoch = {best_epoch} | Final_Acc = {test_acc:.4f}")
    return test_acc

### -------------------------------------------------------- ###

# 写到文件中
def write_to_file(best_epoch, test_acc, args):
    with open('result/basic.txt', 'a', encoding='utf-8') as f:
        f.write(f"lr: {args.lr}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"weight decay: {args.wd}\n")
        f.write("dataset: ")
        if(args.dataset_type == 0):
            f.write("Cora")
        elif(args.dataset_type == 1):
            f.write("Citeseer")
        else:
            f.write("Pubmed")
        f.write("\n")
        f.write(f"Best_epoch = {best_epoch} | Final_Acc = {test_acc:.4f} \n\n")

### -------------------------------------------------------- ###

# 作图部分的函数
def draw(real_epochs, train_loss_history, val_loss_history, acc_history):

    idx = [i for i in range(1,real_epochs+1)]
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # loss
    color_loss = '#FF5733'
    color_val_loss = "#76F81F"
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color_loss, fontsize=12, fontweight='bold')
    loss_line1 = ax1.plot(idx, train_loss_history, color=color_loss, 
                       linewidth=2, alpha=0.8, label='train_loss')
    loss_line2 = ax1.plot(idx, val_loss_history, color=color_val_loss, 
                       linewidth=2, alpha=0.8, label='val_loss')

    ax1.grid(True, axis='y', alpha=0.3)

    # acc
    ax2 = ax1.twinx()
    color_acc = '#3A9BDC'
    ax2.set_ylabel('Acc', color=color_acc, fontsize=12, fontweight='bold')
    acc_line2 = ax2.plot(idx, acc_history, color=color_acc, 
                     linestyle='--', linewidth=2, alpha=0.8, label='acc')

    # 合并
    lines_1 = loss_line1 + loss_line2
    lines_2 = acc_line2
    all_lines = lines_1 + lines_2
    labels = [l.get_label() for l in all_lines]
    plt.legend(all_lines, labels, loc='upper left', framealpha=0.9)

    # 统一标题
    plt.tight_layout()
    # plt.savefig("pic/basic/1.png", dpi=300, bbox_inches='tight')
    plt.show()

### -------------------------------------------------------- ###

def main():
    #解析超参数
    args = parser.parse_args()

    # 读取数据集
    if args.dataset_type == 0:
        dataset = Planetoid(root = "data", name = "Cora")
    elif args.dataset_type == 1:
        dataset = Planetoid(root = "data", name = "Citeseer")
    elif args.dataset_type == 2:
        dataset = Planetoid(root = "data", name = "Pubmed")
    
    data = dataset[0]

    # 取出数据，并且把要计算的部分迁移到cuda上，加速计算
    device = torch.device("cuda")
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)
    val_mask = data.val_mask.to(device)
    num_nodes = data.num_nodes
    num_features = data.num_features
    num_classes = dataset.num_classes

    # 增加自环
    edge_index = data.edge_index.to(device)
    idx = torch.arange(end = num_nodes, dtype = torch.long, device = device)
    self_index = torch.stack([idx, idx], dim = 0)
    edge_index = torch.cat([edge_index, self_index], dim = 1)

    # 计算权重矩阵
    adj = get_adj(edge_index, num_nodes)

    # 训练前的各种定义
    model = GCN(in_feature=num_features, hidden_feature=args.hidden_features, 
                out_feature=num_classes, dropout=args.dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    epochs = args.epochs

    # 早停机制的参数
    patience = args.patience
    best_val_acc = 0
    cnt = 0
    best_model_state = None
    real_epochs = epochs
    best_epoch = 0

    # 绘图记录的list
    # 绘图用
    acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        logits, loss = train_for_one_epoch(model, optimizer, x, adj, y, train_mask, args.wd)
        acc, val_loss = eval_for_one_epoch(model, logits, y, val_mask, args.wd)

        # 数据加入到list里边，画图用
        acc_history.append(acc)
        train_loss_history.append(loss.item())
        val_loss_history.append(val_loss.item())

        # 早停判断
        if(acc > best_val_acc):
            cnt = 0
            best_val_acc = acc
            best_epoch = epoch+1
            best_model_state = copy.deepcopy(model.state_dict())
        elif(cnt >= patience):
            print("early stop!")
            real_epochs = epoch+1
            break
        else:
            cnt += 1
        
        # 打印训练日志
        if(epoch < 20 or (epoch + 1) % 5 == 0 and epoch >= 20):
            print(f"Epoch {epoch+1}\t| Train_Loss: {loss.item():.4f}\t| Val_Loss: {val_loss.item():.4f}\t| Val_acc: {acc:.4f}")

    model.load_state_dict(best_model_state)
    test_acc = final_test(model, x, adj, y, test_mask, best_epoch)
    write_to_file(best_epoch, test_acc, args)
    
    draw(real_epochs, train_loss_history, val_loss_history, acc_history)

if __name__=="__main__":
    main()
