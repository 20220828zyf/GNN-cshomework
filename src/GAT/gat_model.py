import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    图注意力层（Graph Attention Layer）
    实现 GAT 论文中的注意力机制
    
    Paper: "Graph Attention Networks" (Velickovic et al., ICLR 2018)
    
    注意力机制：
    e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    α_ij = softmax_j(e_ij)
    h_i' = σ(Σ_j∈N_i α_ij W h_j)
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: dropout 概率
            alpha: LeakyReLU 的负斜率
            concat: 如果为 True，使用多头注意力的拼接；如果为 False，使用平均
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 线性变换矩阵 W
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # 注意力机制参数 a（用于计算注意力系数）
        # a 是一个 2*out_features 维的向量，用于计算 [Wh_i || Wh_j] 的注意力分数
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot 初始化（Xavier uniform）
        init_range = np.sqrt(6.0 / (self.in_features + self.out_features))
        nn.init.uniform_(self.W, -init_range, init_range)
        
        # 注意力参数 a 使用较小的初始化
        nn.init.xavier_uniform_(self.a)

    def forward(self, input, adj):
        """
        Args:
            input: 节点特征矩阵 [N, in_features]
            adj: 邻接矩阵 [N, N]（可以是稀疏或密集）
        Returns:
            输出特征矩阵 [N, out_features] 或 [N, out_features * num_heads]（如果 concat=True）
        """
        # 线性变换: h' = Wh
        h = torch.mm(input, self.W)  # [N, out_features]
        N = h.size(0)
        
        # 计算注意力系数
        # 对于每对节点 (i, j)，计算 e_ij = a^T [Wh_i || Wh_j]
        # 使用更高效的方式：a^T [h_i || h_j] = a_left^T h_i + a_right^T h_j
        
        # 将 a 拆分为两部分：a_left 和 a_right
        a_left, a_right = self.a.split(self.out_features, 0)
        # a_left: [out_features, 1]
        # a_right: [out_features, 1]
        
        # 计算 a_left^T h_i 和 a_right^T h_j
        # h: [N, out_features]
        # a_left: [out_features, 1]
        # a_right: [out_features, 1]
        attention_left = torch.matmul(h, a_left)  # [N, out_features] @ [out_features, 1] = [N, 1]
        attention_right = torch.matmul(h, a_right)  # [N, out_features] @ [out_features, 1] = [N, 1]
        
        # 使用广播机制计算所有节点对的注意力分数
        # e_ij = a_left^T h_i + a_right^T h_j
        # attention_left: [N, 1], attention_right: [N, 1]
        # 广播: [N, 1] + [1, N] = [N, N]
        e = attention_left + attention_right.t()  # [N, 1] + [1, N] = [N, N]
        
        # LeakyReLU 激活
        e = F.leaky_relu(e, negative_slope=self.alpha)
        
        # 应用邻接矩阵掩码：只保留有边的节点对的注意力
        # 将没有边的节点对的注意力分数设为负无穷（softmax 后会变为 0）
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        # 直接修改 e 而不是创建副本（节省内存）
        e[adj_dense == 0] = -9e15
        
        # Softmax 归一化：α_ij = softmax_j(e_ij)
        attention = F.softmax(e, dim=1)
        
        # 释放中间变量以节省内存
        if adj.is_sparse:
            del adj_dense
        del e
        
        # Dropout
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 计算输出：h_i' = Σ_j α_ij h_j
        h_prime = torch.matmul(attention, h)  # [N, out_features]
        
        # 如果 concat=True，直接返回（用于多头注意力的拼接）
        # 如果 concat=False，返回（用于多头注意力的平均）
        if self.concat:
            return F.elu(h_prime)  # ELU 激活
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    """
    图注意力网络（Graph Attention Network）
    
    结构：
    - 第一层：多头图注意力层（input_dim -> hidden_dim，K 个头）
    - 第二层：单头图注意力层（hidden_dim * K -> output_dim，1 个头）
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        """
        Args:
            nfeat: 输入特征维度
            nhid: 隐藏层维度
            nclass: 输出类别数
            dropout: dropout 概率
            alpha: LeakyReLU 的负斜率
            nheads: 第一层的注意力头数
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # 第一层：多头注意力（拼接）
        # 每个头输出 nhid 维，K 个头拼接后为 nhid * K 维
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ])
        
        # 第二层：单头注意力（平均）
        # 输入维度是 nhid * nheads（因为第一层是拼接的）
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        """
        Args:
            x: 节点特征矩阵 [N, nfeat]
            adj: 邻接矩阵 [N, N]（可以是稀疏或密集）
        Returns:
            输出 logits [N, nclass]（不包含 softmax）
        """
        # 如果输入是稀疏的，转换为密集矩阵
        if x.is_sparse:
            x = x.to_dense()
        
        # 第一层：多头注意力
        # 每个头独立计算，然后拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # [N, nhid * nheads]
        
        # 第二层：单头注意力
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)  # [N, nclass]
        
        return x  # 返回 logits，不包含 softmax
    
    def predict(self, x, adj):
        """
        预测函数，返回 softmax 概率
        
        Args:
            x: 节点特征矩阵 [N, nfeat]
            adj: 邻接矩阵 [N, N]
        Returns:
            softmax 概率 [N, nclass]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, adj)
            return F.softmax(outputs, dim=1)

