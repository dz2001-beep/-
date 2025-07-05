import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import remove_self_loops, add_self_loops


class SFC_GCN_Layer(nn.Module):
    def __init__(self, in_features, out_features, gamma):
        super(SFC_GCN_Layer, self).__init__()
        # 输入特征的维度
        self.in_features = in_features
        # 输出特征的维度
        self.out_features = out_features
        """gamma"""
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=True)
        # 定义可学习的权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 定义可学习的偏置向量
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 使用Xavier初始化权重矩阵
        nn.init.xavier_uniform_(self.weight)
        # 将偏置向量初始化为0
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, S = None):
        # x: 节点特征矩阵，大小为[num_nodes, in_features]
        # edge_index: 边索引，大小为[2, num_edges]
        gamma = torch.sigmoid(self.gamma)
        # 计算规范化的邻接矩阵
        num_nodes = x.size(0)
        """1. 先去掉自环，再重新添加自环"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       edge_attr=None,
                                       fill_value=1.0,
                                       num_nodes=num_nodes)

        # 创建一个全零矩阵作为邻接矩阵的初始状态
        adj = torch.zeros(num_nodes, num_nodes).cuda()
        # 根据edge_index填充邻接矩阵，无向图因此两个方向都要填充
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
        '''在自环的邻接上注入空间功能一致性得分 S'''
        if S is not None:
            # 乘性重调：A_ij ← A_ij * (1 + γ·S_ij)
            adj = adj * (1 + gamma * S)

        # 计算每个节点的度
        deg = torch.sum(adj, dim=1)
        # 计算度矩阵的逆平方根，用于后续的归一化
        deg_inv_sqrt = 1.0 / torch.sqrt(deg)  # 加上一个小的常数避免除零错误
        # 计算对称归一化的邻接矩阵
        norm_adj = adj * deg_inv_sqrt.unsqueeze(1)
        norm_adj = norm_adj * deg_inv_sqrt.unsqueeze(0)

        # 支撑传播：线性变换节点特征
        support = torch.matmul(x, self.weight)  # X * W
        # 消息传递：通过归一化的邻接矩阵传播特征
        output = torch.matmul(norm_adj, support) + self.bias  # D^{-1/2} * A * D^{-1/2} * (X * W) + b
        return output  # 返回输出，可以选择使用ReLU等激活函数进行非线性变换



'''SFC_GCN模块'''
class SFC_GCN(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers, gamma):
        super(SFC_GCN, self).__init__()
        self.gcn_layers = gcn_layers
        self.dropout = dropout

        self.gcns = nn.ModuleList([
            SFC_GCN_Layer(embedding_size, embedding_size, gamma)
            for _ in range(self.gcn_layers)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(embedding_size)
            for _ in range(self.gcn_layers - 1)
        ])

    def forward(self, features, edge_index, S_dict, is_training=True):
        n_emb = poi_emb = s_emb = d_emb = features
        poi_edge_index, s_edge_index, d_edge_index, n_edge_index = edge_index
        S_n, S_poi, S_s, S_d = S_dict['n'], S_dict['poi'], S_dict['s'], S_dict['d']

        for i in range(self.gcn_layers - 1):
            # 1. 地理邻居视图
            tmp = n_emb
            n_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](n_emb, n_edge_index, S_n)))
            if is_training:
                n_emb = F.dropout(n_emb, p=self.dropout)

            # 2. poi视图
            tmp = poi_emb
            poi_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](poi_emb, poi_edge_index, S_poi)))
            if is_training:
                poi_emb = F.dropout(poi_emb, p=self.dropout)

            # 3. 流入视图
            tmp = s_emb
            s_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](s_emb, s_edge_index, S_s)))
            if is_training:
                s_emb = F.dropout(s_emb, p=self.dropout)

            # 4. 流出视图
            tmp = d_emb
            d_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](d_emb, d_edge_index, S_d)))
            if is_training:
                d_emb = F.dropout(d_emb, p=self.dropout)

        n_emb = self.gcns[-1](n_emb, n_edge_index, S_n)
        poi_emb = self.gcns[-1](poi_emb, poi_edge_index, S_poi)
        s_emb = self.gcns[-1](s_emb, s_edge_index, S_s)
        d_emb = self.gcns[-1](d_emb, d_edge_index, S_d)

        return n_emb, poi_emb, s_emb, d_emb
