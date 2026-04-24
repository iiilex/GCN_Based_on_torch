import model
import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import pickle
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt

def get_data_by_hand(args):
    root = "" 
    if args.dataset_type == 0:
        root = "./data/Cora/raw/ind.cora."
    elif args.dataset_type == 1:
        root = "./data/Citeseer/raw/ind.citeseer."
    elif args.dataset_type == 2:
        root = "./data/Pubmed/raw/ind.pubmed."
    
    with open(root + "allx", "rb") as f:
        allx = pickle.load(f, encoding='latin1')

    with open(root + "ally", "rb") as f:
        ally = pickle.load(f, encoding='latin1')

    with open(root + "graph", "rb") as f:
        graph = pickle.load(f, encoding='latin1')
    
    test_index = []
    with open(root + "test.index", "rb") as f:
        test_index.extend(int(line.strip()) for line in f)
    test_index_sorted = sorted(test_index)
    
    with open(root + "tx", "rb") as f:
        tx = pickle.load(f, encoding='latin1')

    with open(root + "ty", "rb") as f:
        ty = pickle.load(f, encoding='latin1')

    with open(root + "x", "rb") as f:
        x = pickle.load(f, encoding='latin1')

    with open(root + "y", "rb") as f:
        y = pickle.load(f, encoding='latin1')

    # 由原论文，划分方法如下：
    # 按顺序把 allx 和 tx 拼起来。
    # allx的前(x的长度)个是训练集
    # 接着的500个是验证集
    # allx + tx 拼起来
    # 剩下的是不参与训练或验证，只参加信息传递的

    if(args.dataset_type == 1):
        test_index_full = range(min(test_index), max(test_index)+1)

        tx_extended = sparse.lil_matrix((len(test_index_full), x.shape[1]))
        tx_extended[test_index - min(test_index), :] = tx
        tx = tx_extended

        ty_extended = sparse.lil_matrix((len(test_index_full), y.shape[1]))
        ty_extended[test_index - min(test_index), :] = ty
        ty = ty_extended

    # 得到一些数据集基本数据
    num_nodes = allx.shape[0] + tx.shape[0]
    num_features = x.shape[1]
    num_classes = y.shape[1]

    # 训练、验证、测试集掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype = torch.bool)

    train_mask[:x.shape[0]] = True
    val_mask[x.shape[0]:x.shape[0]+500] = True
    test_mask[test_index] = True

    # 按照索引进行重排，防止图的关系错位了
    features = sparse.vstack((allx, tx))
    features = features.tolil()

    labels = sparse.vstack((ally, ty))
    labels = labels.tolil()

    features[test_index, :] = features[test_index_sorted, :]
    labels[test_index, :] = labels[test_index_sorted, :]

    # 把读入的csr_matrix 转换为运算用的 coo_tensor
    def csr_to_coo_tensor(t):
        t = t.tocoo()
        index = np.vstack((t.row, t.col))
        index = torch.from_numpy(index).long()
        value = torch.from_numpy(t.data).float()
        shape = torch.Size(t.shape)
        return torch.sparse_coo_tensor(index, value, shape)

    features = csr_to_coo_tensor(features)
    labels = csr_to_coo_tensor(labels)
    labels = labels.to_dense()
    labels = labels.argmax(dim = 1)

    # 得到 edge_index
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj.tocoo()
    edge_index = np.vstack((adj.row, adj.col))
    edge_index = torch.from_numpy(edge_index).long()

    # 全部移动到gpu
    device = torch.device("cuda")
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)

    return device, features, labels, train_mask, test_mask, val_mask, num_nodes, num_features, num_classes, edge_index

# 获取数据集
def get_data(args):
    if args.dataset_type == 0:
        dataset = Planetoid(root = "data", name = "Cora")
    elif args.dataset_type == 1:
        dataset = Planetoid(root = "data", name = "Citeseer")
    elif args.dataset_type == 2:
        dataset = Planetoid(root = "data", name = "Pubmed")

    data = dataset[0]

    # 取出数据，并且把要计算的部分迁移到cuda上，加速计算
    device = torch.device("cuda")
    x = data.x.to(device) # 原始数据矩阵
    y = data.y.to(device) # 标签
    train_mask = data.train_mask.to(device) # 训练集掩码
    test_mask = data.test_mask.to(device) # 测试集掩码
    val_mask = data.val_mask.to(device) # 验证集掩码
    num_nodes = data.num_nodes # 节点数量
    num_features = data.num_features # 每个节点的特征数量
    num_classes = dataset.num_classes # 分类数（标签种类数）
    edge_index = data.edge_index.to(device) # 边的信息
    return device, x, y, train_mask, test_mask, val_mask, num_nodes, num_features, num_classes, edge_index

# 计算权重矩阵
def get_adj(edge_index, num_nodes, device): 

    # 增加自环
    idx = torch.arange(end = num_nodes, dtype = torch.long, device = device)
    self_index = torch.stack([idx, idx], dim = 0)
    edge_index = torch.cat([edge_index, self_index], dim = 1)

    # 计算 D^{-1/2}
    u, v = edge_index
    D = torch.bincount(u, minlength=num_nodes).float()
    inv_sqrt_D = D.pow(-0.5)
    inv_sqrt_D[inv_sqrt_D == float("inf")] = 0

    # 在实际代码中，归一化邻接矩阵的值以边权的形式存在，来配合稀疏矩阵的形式
    edge_weight = inv_sqrt_D[u] * inv_sqrt_D[v]
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    return adj

# 根据参数选取模型
def get_model(args, in_feature, out_feature):
    if args.model_type == 0:
        return model.MLP_2(in_feature, args.hidden_feature, 
                out_feature, dropout=args.dropout)
    elif args.model_type == 1:
        return model.GCN_2(in_feature, args.hidden_feature, 
                out_feature, dropout=args.dropout)
    elif args.model_type == 2:
        return model.GCN_8(in_feature, args.hidden_feature, 
                out_feature, dropout=args.dropout)
    elif args.model_type == 3:
        return model.ResGCN_8(in_feature, args.hidden_feature, 
                out_feature, dropout=args.dropout)
    
def print_in_epoch(train_acc, val_acc, train_loss, val_loss, epoch):
    print(f"Epoch {epoch+1}\t| Train_Loss: {train_loss:.3f}\t| Val_Loss: {val_loss:.3f}\t| Train_acc: {train_acc:.3f}\t| Val_acc: {val_acc:.3f}")

def write_to_file(best_epoch, test_acc, args):
    path = ""
    if args.model_type == 0:
        path = './result/MLP_2.txt'
    elif args.model_type == 1:
        path = './result/GCN_2.txt'
    elif args.model_type == 2:
        path = './result/GCN_8.txt'
    elif args.model_type == 3:
        path = './result/ResGCN_8.txt'

    with open(path, 'a', encoding='utf-8') as f:
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

def draw(real_epochs, train_loss_history, val_loss_history, train_acc_history, val_acc_history):

    idx = [i for i in range(1,real_epochs+1)]
    train_loss_history = train_loss_history[:real_epochs]
    val_loss_history = val_loss_history[:real_epochs]
    train_acc_history = train_acc_history[:real_epochs]
    val_acc_history = val_acc_history[:real_epochs]
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # loss
    color_loss = '#FF5733'
    color_val_loss = "#EF8325"
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color_loss, fontsize=12, fontweight='bold')
    loss_line1 = ax1.plot(idx, train_loss_history, color=color_loss, 
                       linewidth=2, alpha=0.8, label='train_loss')
    loss_line2 = ax1.plot(idx, val_loss_history, color=color_val_loss, 
                       linewidth=2, alpha=0.8, label='val_loss')

    ax1.grid(True, axis='y', alpha=0.3)

    # acc
    ax2 = ax1.twinx()
    color_val_acc = '#3A9BDC'
    color_train_acc = "#3ADCCF"
    ax2.set_ylabel('Acc', color=color_val_acc, fontsize=12, fontweight='bold')
    acc_line1 = ax2.plot(idx, train_acc_history, color=color_train_acc, 
                     linestyle='--', linewidth=2, alpha=0.8, label='train_acc')
    acc_line2 = ax2.plot(idx, val_acc_history, color=color_val_acc, 
                     linestyle='--', linewidth=2, alpha=0.8, label='val_acc')

    # 合并
    lines_1 = loss_line1 + loss_line2
    lines_2 = acc_line1 + acc_line2
    all_lines = lines_1 + lines_2
    labels = [l.get_label() for l in all_lines]
    plt.legend(all_lines, labels, loc='upper left', framealpha=0.9)

    # 统一标题
    plt.tight_layout()
    # plt.savefig("pic/basic/1.png", dpi=300, bbox_inches='tight')
    plt.show()