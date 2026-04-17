基于论文 Semi-Supervised Classification with Graph Convolutional Networks 进行的复现实验。

GCN.py : 2层标准模型

Dense.py : 2层MLP(baseline)

版本:

python : 3.11.17

pytorch : 2.9.1+cu128

torch_geometric : 2.7.0

尽可能还原了原版的计算过程。

食用方法:
1. 准备尽可能相同的环境
2. 在该根目录下打开powershell
3. 输入 conda init powershell(假如已经init过了就不用，如果是第一次init记得重新打开)
4. 输入 python [文件名字] -h 获取帮助，或者直接输入 python [文件名字] 以默认参数运行