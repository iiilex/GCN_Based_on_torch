基于论文 Semi-Supervised Classification with Graph Convolutional Networks 进行的复现实验。

GCN_basic.py 对应标准模型

GCN_res_n.py 对应n层的残差连接模型

版本:

python: 3.11.17

pytorch: 2.9.1+cu128

torch_geometric: 2.7.0

注:由于原论文使用的MXNet太过古老，于是使用torch进行复现。尽可能还原了原版的计算过程。

作图部分的代码为AI生成后自行修改