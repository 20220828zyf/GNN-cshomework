import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """
    图卷积层（Graph Convolution Layer）
    对应原始 TensorFlow 实现中的 GraphConvolution 层
    
    支持标准 GCN 和 Chebyshev 多项式版本的 GCN
    
    标准 GCN: Z = A_hat X W
    其中 A_hat = D^{-1/2} (A + I) D^{-1/2} 是归一化的邻接矩阵
    
    Chebyshev GCN: Z = sum_k T_k(L) X W_k
    其中 T_k(L) 是 Chebyshev 多项式，L 是拉普拉斯矩阵
    """
    def __init__(self, in_features, out_features, num_supports=1, bias=False, sparse_inputs=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_supports = num_supports
        self.sparse_inputs = sparse_inputs
        
        # 为每个 support 创建权重矩阵（对应原始实现中的 weights_0, weights_1, ...）
        # 对于标准 GCN: num_supports = 1
        # 对于 Chebyshev GCN: num_supports = max_degree + 1
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(num_supports)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot 初始化（Xavier uniform），对应原始实现中的 glorot 初始化
        init_range = np.sqrt(6.0 / (self.in_features + self.out_features))
        for weight in self.weights:
            nn.init.uniform_(weight, -init_range, init_range)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, support, dropout=0.0):
        """
        Args:
            input: 节点特征矩阵 [N, in_features]（可以是稀疏或密集张量）
            support: 邻接矩阵列表（对于标准 GCN 是长度为 1 的列表，对于 Chebyshev 是长度为 k+1 的列表）
            dropout: dropout 概率
        Returns:
            输出特征矩阵 [N, out_features]
        """
        x = input
        
        # 如果输入是稀疏的，转换为密集矩阵（PyTorch 稀疏张量不支持直接与密集矩阵相乘）
        if x.is_sparse:
            x = x.to_dense()
        
        # Dropout（对应原始实现中的 dropout）
        if dropout > 0.0 and self.training:
            x = F.dropout(x, p=dropout, training=self.training)
        
        # 对每个 support 计算卷积并求和
        # 对应原始实现: supports = [dot(support[i], dot(x, weights[i])) for i in range(num_supports)]
        # output = sum(supports)
        supports = []
        for i in range(len(support)):
            adj = support[i]
            weight = self.weights[i]
            
            # 图卷积操作: pre_sup = XW, support_output = A_hat * pre_sup
            pre_sup = torch.mm(x, weight)  # [N, out_features]
            
            # 使用稀疏矩阵乘法（如果 adj 是稀疏的）
            if adj.is_sparse:
                support_output = torch.spmm(adj, pre_sup)  # [N, out_features]
            else:
                support_output = torch.mm(adj, pre_sup)  # [N, out_features]
            
            supports.append(support_output)
        
        # 对所有 support 的输出求和（对应原始实现中的 tf.add_n）
        output = torch.stack(supports, dim=0).sum(dim=0)  # [N, out_features]
        
        # 添加偏置（如果存在）
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GCN(nn.Module):
    """
    两层 GCN 模型
    对应原始 TensorFlow 实现中的 GCN 类
    
    支持标准 GCN 和 Chebyshev 多项式版本的 GCN
    
    结构：
    - 第一层: GraphConvolution(input_dim -> hidden1, ReLU激活, dropout=True)
    - 第二层: GraphConvolution(hidden1 -> output_dim, 无激活, dropout=True)
    """
    def __init__(self, nfeat, nhid, nclass, num_supports=1, dropout=0.5, sparse_inputs=True):
        """
        Args:
            nfeat: 输入特征维度
            nhid: 隐藏层维度（对应 FLAGS.hidden1）
            nclass: 输出类别数
            num_supports: support 的数量
                - 对于标准 GCN: num_supports = 1
                - 对于 Chebyshev GCN: num_supports = max_degree + 1
            dropout: dropout 概率
            sparse_inputs: 是否使用稀疏输入
        """
        super(GCN, self).__init__()
        
        # 第一层：input_dim -> hidden1，使用 ReLU 激活，有 dropout
        self.gc1 = GraphConvolution(nfeat, nhid, num_supports=num_supports, 
                                    bias=False, sparse_inputs=sparse_inputs)
        
        # 第二层：hidden1 -> output_dim，无激活函数，有 dropout
        self.gc2 = GraphConvolution(nhid, nclass, num_supports=num_supports, 
                                    bias=False, sparse_inputs=False)
        
        self.dropout = dropout

    def forward(self, x, support):
        """
        Args:
            x: 节点特征矩阵 [N, nfeat]
            support: 邻接矩阵列表
                - 对于标准 GCN: [adj] (长度为 1 的列表)
                - 对于 Chebyshev GCN: [T_0(L), T_1(L), ..., T_k(L)] (长度为 k+1 的列表)
        Returns:
            输出 logits [N, nclass]（不包含 softmax，对应原始实现的 outputs）
        """
        # 第一层：GraphConvolution + ReLU + Dropout
        x = self.gc1(x, support, dropout=self.dropout)
        x = F.relu(x)  # ReLU 激活
        
        # 第二层：GraphConvolution + Dropout（无激活函数）
        x = self.gc2(x, support, dropout=self.dropout)
        # 注意：原始实现中第二层使用 lambda x: x（无激活函数）
        
        return x  # 返回 logits，不包含 softmax
    
    def predict(self, x, support):
        """
        预测函数，返回 softmax 概率
        对应原始实现中的 predict 方法
        
        Args:
            x: 节点特征矩阵 [N, nfeat]
            support: 邻接矩阵列表
        Returns:
            softmax 概率 [N, nclass]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, support)
            return F.softmax(outputs, dim=1)
