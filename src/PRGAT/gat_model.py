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
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True, 
                 use_pagerank=False, pagerank_mode='high_to_low', pagerank_alpha=0.85):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: dropout 概率
            alpha: LeakyReLU 的负斜率
            concat: 如果为 True，使用多头注意力的拼接；如果为 False，使用平均
            use_pagerank: 是否使用 PageRank 分数调整注意力
            pagerank_mode: PageRank 使用模式
                - 'high_to_low': PageRank 分数高的节点归一化值高（默认）
                - 'low_to_high': PageRank 分数低的节点归一化值高
                - 'middle_to_high': PageRank 分数处于中等的节点归一化值高
            pagerank_alpha: PageRank 的阻尼因子
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.use_pagerank = use_pagerank
        self.pagerank_mode = pagerank_mode
        self.pagerank_alpha = pagerank_alpha
        
        # 线性变换矩阵 W
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # 注意力机制参数 a（用于计算注意力系数）
        # a 是一个 2*out_features 维的向量，用于计算 [Wh_i || Wh_j] 的注意力分数
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        
        self.reset_parameters()

    def pagerank(self, adj, alpha=0.85, max_iter=100, tol=1e-6):
        """
        Args:
            adj: 邻接矩阵 [N, N]（可以是稀疏或密集）
            alpha: 阻尼因子
            max_iter: 最大迭代次数
            tol: 收敛阈值
        Returns:
            PageRank分数向量 [N]
        """
        # 转换为密集矩阵以便计算
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        N = adj_dense.shape[0]
    
        # 1. 计算转移矩阵
        d = adj_dense.sum(dim=1)  # 每个节点的出度 [N]
        d_inv = 1.0 / d
        d_inv[d == 0] = 0  # 避免除零，悬挂节点设为0
        M = adj_dense * d_inv.unsqueeze(1)  # 转移矩阵 [N, N]
    
        # 2. 处理悬挂节点（出度为0的节点）
        # 在标准PageRank中，这些节点应该以等概率跳转到所有节点
        dangling_nodes = (d == 0)
        if dangling_nodes.any():
            # 悬挂节点的转移概率设为1/N（每行的所有元素都设为1/N）
            M[dangling_nodes, :] = 1.0 / N
    
        # 3. 加入阻尼因子和随机跳转
        # PageRank公式: π = (1-α)/N * e + α * M^T * π
        # 其中e是全1向量，所以 (1-α)/N * e 是每个元素都是 (1-α)/N 的向量
        
        # 初始化为均匀分布 [N]
        pagerank = torch.ones(N, device=adj_dense.device, dtype=adj_dense.dtype) / N
        
        # 迭代计算PageRank
        for _ in range(max_iter):
            # M^T * π: [N, N] @ [N, 1] = [N, 1]
            M_transpose_pi = torch.mm(M.t(), pagerank.unsqueeze(1)).squeeze(1)  # [N]
            
            # PageRank更新公式
            new_pagerank = (1-alpha) / N + alpha * M_transpose_pi  # [N]
            
            # 检查收敛
            if torch.norm(new_pagerank - pagerank) < tol:
                break
            pagerank = new_pagerank
        
        return pagerank  # 返回 [N] 维的PageRank分数向量

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
        
        # 如果使用 PageRank，调整注意力分数
        if self.use_pagerank:
            # 计算 PageRank 分数
            pagerank_scores = self.pagerank(adj, alpha=self.pagerank_alpha, max_iter=100, tol=1e-6)
            # pagerank_scores: [N]
            
            # 创建 PageRank 调整矩阵 [N, N]
            # 对于每个节点对 (i, j)，如果 j 是 i 的邻居，则使用 j 的 PageRank 分数
            pr_matrix = pagerank_scores.unsqueeze(0).expand(N, N)  # [N, N]，每行都是相同的 PageRank 分数
            
            # 只保留邻居节点的 PageRank 分数，非邻居设为很小的值（用于 softmax）
            # 对于非邻居节点，设置为很小的值，这样 softmax 后会接近 0
            epsilon = 1e-8
            pr_matrix = pr_matrix * adj_dense + epsilon * (1 - adj_dense)  # [N, N]
            
            if self.pagerank_mode == 'high_to_low':
                # 方式1：PageRank 分数高的节点归一化值高
                # 直接对 PageRank 分数进行 softmax 归一化
                pr_normalized = F.softmax(pr_matrix, dim=1)  # [N, N]
            elif self.pagerank_mode == 'low_to_high':
                # 方式2：PageRank 分数低的节点归一化值高
                # 对 PageRank 分数的倒数进行 softmax 归一化
                inverted_pr = 1.0 / (pr_matrix + epsilon)  # [N, N]
                # 对于非邻居节点，将倒数设为很小的值
                inverted_pr = inverted_pr * adj_dense + epsilon * (1 - adj_dense)
                pr_normalized = F.softmax(inverted_pr, dim=1)  # [N, N]
            else:  # 'middle_to_high'
                # 方式3：PageRank 分数处于中等的节点归一化值高
                # 对于每个节点 i，计算其邻居节点 PageRank 分数的中位数
                # 然后使用高斯函数：权重 = exp(-(pr - median)^2 / variance)
                pr_normalized = torch.zeros_like(pr_matrix)  # [N, N]
                
                for i in range(N):
                    # 获取节点 i 的邻居节点索引
                    neighbors = adj_dense[i] > 0  # [N] bool
                    if neighbors.any():
                        neighbor_pr_scores = pr_matrix[i][neighbors]  # [num_neighbors]
                        
                        # 计算中位数
                        median_pr = torch.median(neighbor_pr_scores)
                        
                        # 计算标准差（用于归一化距离）
                        std_pr = torch.std(neighbor_pr_scores)
                        if std_pr < epsilon:
                            std_pr = torch.tensor(1.0, device=std_pr.device, dtype=std_pr.dtype)
                        
                        # 计算每个邻居节点与中位数的距离（使用高斯函数）
                        # 距离中位数越近，权重越大
                        distances = torch.abs(neighbor_pr_scores - median_pr)  # [num_neighbors]
                        # 使用高斯函数：exp(-distance^2 / (2 * std^2))
                        # 距离中位数越近，exp 值越大
                        gaussian_weights = torch.exp(-distances.pow(2) / (2 * std_pr.pow(2) + epsilon))  # [num_neighbors]
                        
                        # 对于非邻居节点，权重设为很小的值
                        pr_normalized[i][neighbors] = gaussian_weights
                        pr_normalized[i][~neighbors] = epsilon
                
                # 对每行进行 softmax 归一化
                pr_normalized = F.softmax(pr_normalized, dim=1)  # [N, N]
            
            # 将归一化的 PageRank 分数与注意力分数相乘
            attention = attention * pr_normalized
            
            # 重新归一化，确保每行的和仍为 1
            attention = attention / (attention.sum(dim=1, keepdim=True) + epsilon)
        
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
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8,
                 use_pagerank=False, pagerank_mode='high_to_low', pagerank_alpha=0.85):
        """
        Args:
            nfeat: 输入特征维度
            nhid: 隐藏层维度
            nclass: 输出类别数
            dropout: dropout 概率
            alpha: LeakyReLU 的负斜率
            nheads: 第一层的注意力头数
            use_pagerank: 是否使用 PageRank 分数调整注意力
            pagerank_mode: PageRank 使用模式
                - 'high_to_low': PageRank 分数高的节点归一化值高（默认）
                - 'low_to_high': PageRank 分数低的节点归一化值高
                - 'middle_to_high': PageRank 分数处于中等的节点归一化值高
            pagerank_alpha: PageRank 的阻尼因子
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # 第一层：多头注意力（拼接）
        # 每个头输出 nhid 维，K 个头拼接后为 nhid * K 维
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True,
                              use_pagerank=use_pagerank, pagerank_mode=pagerank_mode, 
                              pagerank_alpha=pagerank_alpha)
            for _ in range(nheads)
        ])
        
        # 第二层：单头注意力（平均）
        # 输入维度是 nhid * nheads（因为第一层是拼接的）
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False,
                                         use_pagerank=use_pagerank, pagerank_mode=pagerank_mode,
                                         pagerank_alpha=pagerank_alpha)

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

