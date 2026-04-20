import utils
import argparse
import torch
import torch.optim as optim
import copy
#定义所有需要的参数
parser = argparse.ArgumentParser(description="超参数")
parser.add_argument("--hidden_feature", type=int, default=16, help="隐藏层特征数")
parser.add_argument("--lr", type=float, default=0.01, help="学习率")
parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout率")
parser.add_argument("--wd", type=float, default=5e-4, help="权重衰减")
parser.add_argument("--patience", type=int, default=30, help="早停轮数")
parser.add_argument("--dataset_type", type=int, default=0, help="0为Cora, 1为Citeseer, 2为Pubmed, 默认Cora")
parser.add_argument("--model_type", type=int, default=1, help="0为2层MLP, 1为2层GCN, 2为8层GCN, 3为8层Res-GCN")

# 1个epoch的训练
def train_for_one_epoch(model, optimizer, x, adj, y, train_mask, weight_decay): # 一次训练
    model.train()
    optimizer.zero_grad()
    logits = model(x, adj)
    loss = model.loss_fn(logits, y, train_mask, weight_decay)
    loss.backward()
    optimizer.step()
    return logits, loss

# 1个epoch的评估
def eval_for_one_epoch(model, logits, y, train_mask, val_mask, weight_decay): # 一次评估
    model.eval()
    with torch.no_grad():
        train_acc = model.accuracy_fn(logits, y, train_mask)
        val_acc = model.accuracy_fn(logits, y, val_mask)
        val_loss = model.loss_fn(logits, y, val_mask, weight_decay)
    return train_acc, val_acc, val_loss

# 用最好的模型结果来计算测试集
def final_test(model, x, adj, y, test_mask, best_epoch):
    model.eval()
    with torch.no_grad():
        logits = model(x, adj)
        test_acc = model.accuracy_fn(logits, y, test_mask)
        print(f"Best_epoch = {best_epoch} | Final_Acc = {test_acc:.3f}")
    return test_acc

def main():
    # 获取所需参数
    args = parser.parse_args()
    
    # 通过读取模块获得数据
    device, x, y, train_mask, test_mask, val_mask, num_nodes, num_features, num_classes, edge_index = utils.get_data(args)

    # 计算聚合矩阵
    adj = utils.get_adj(edge_index, num_nodes, device)

    # 获取当前模型
    current_model = utils.get_model(args, num_features, num_classes)
    current_model.to(device)
    optimizer = optim.Adam(current_model.parameters(), lr = args.lr)
    epochs = args.epochs
    
    # 早停的参数
    patience = args.patience
    best_val_acc = 0
    cnt = 0
    best_model_state = None
    real_epochs = epochs
    best_epoch = 0
        
    # 训练过程的记录
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    # 训练循环
    for epoch in range(epochs):
        logits, loss = train_for_one_epoch(current_model, optimizer, x, adj, y, train_mask, args.wd)
        train_acc, val_acc, val_loss = eval_for_one_epoch(current_model, logits, y, train_mask, val_mask, args.wd)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(loss.item())
        val_loss_history.append(val_loss.item())

        # 早停判断
        if(val_acc > best_val_acc):
            cnt = 0
            best_val_acc = val_acc
            best_epoch = epoch+1
            best_model_state = copy.deepcopy(current_model.state_dict())
        elif(cnt >= patience):
            print("early stop!")
            real_epochs = epoch+1
            break
        else:
            cnt += 1
        
        # 打印参数
        if epoch < 9 or (epoch+1) % 10 == 0:
            utils.print_in_epoch(train_acc, val_acc, loss, val_loss, epoch)

    current_model.load_state_dict(best_model_state)
    test_acc = final_test(current_model, x, adj, y, test_mask, best_epoch)
    utils.write_to_file(best_epoch, test_acc, args)
    utils.draw(real_epochs, train_loss_history, val_loss_history, train_acc_history, val_acc_history)
    







if __name__ == "__main__":
    main()