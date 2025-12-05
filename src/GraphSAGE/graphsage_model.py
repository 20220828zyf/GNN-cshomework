import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class MeanAggregator(nn.Module):
    """
    平均聚合器（Mean Aggregator）
    对邻居特征进行平均聚合
    
    GraphSAGE 论文中的实现：
    h_v^k = σ(W^k · CONCAT(h_v^{k-1}, AGG({h_u^{k-1}, ∀u ∈ N(v)})))
    其中 AGG 是平均聚合
    """
    def __init__(self, in_features, out_features, use_bias=False):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            use_bias: 是否使用偏置
        """
        super(MeanAggregator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 线性层：用于转换拼接后的特征 [self_features || neigh_features]
        # 输入维度是 2 * in_features（拼接自身和邻居特征）
        self.linear = nn.Linear(2 * in_features, out_features, bias=use_bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot 初始化
        init_range = np.sqrt(6.0 / (2 * self.in_features + self.out_features))
        nn.init.uniform_(self.linear.weight, -init_range, init_range)
        
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, self_features, neigh_features):
        """
        Args:
            self_features: 自身特征 [batch_size, in_features]
            neigh_features: 聚合后的邻居特征 [batch_size, in_features]
        Returns:
            输出特征 [batch_size, out_features]
        """
        # 拼接自身特征和邻居特征
        combined = torch.cat([self_features, neigh_features], dim=1)  # [batch_size, 2*in_features]
        
        # 通过线性层
        output = self.linear(combined)  # [batch_size, out_features]
        
        return output


class MaxPoolAggregator(nn.Module):
    """
    最大池化聚合器（MaxPool Aggregator）
    对邻居特征进行最大池化聚合
    
    GraphSAGE 论文中的实现：
    先对每个邻居特征应用 MLP，然后进行最大池化
    """
    def __init__(self, in_features, out_features, use_bias=False):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            use_bias: 是否使用偏置
        """
        super(MaxPoolAggregator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # MLP 用于邻居特征变换（在池化之前）
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )
        
        # 线性层：用于转换拼接后的特征 [self_features || pooled_neigh_features]
        self.linear = nn.Linear(in_features + out_features, out_features, bias=use_bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        # MLP 初始化
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # 线性层初始化
        init_range = np.sqrt(6.0 / (self.in_features + self.out_features + self.out_features))
        nn.init.uniform_(self.linear.weight, -init_range, init_range)
        
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, self_features, neigh_features_list):
        """
        Args:
            self_features: 自身特征 [batch_size, in_features]
            neigh_features_list: 邻居特征列表 [batch_size, num_neighbors, in_features]
        Returns:
            输出特征 [batch_size, out_features]
        """
        # 对每个邻居应用 MLP
        batch_size, num_neighbors, _ = neigh_features_list.shape
        neigh_reshaped = neigh_features_list.view(-1, self.in_features)  # [batch_size * num_neighbors, in_features]
        neigh_transformed = self.mlp(neigh_reshaped)  # [batch_size * num_neighbors, out_features]
        neigh_transformed = neigh_transformed.view(batch_size, num_neighbors, self.out_features)
        
        # 最大池化
        neigh_pooled = torch.max(neigh_transformed, dim=1)[0]  # [batch_size, out_features]
        
        # 拼接自身特征和池化后的邻居特征
        combined = torch.cat([self_features, neigh_pooled], dim=1)  # [batch_size, in_features + out_features]
        
        # 通过线性层
        output = self.linear(combined)  # [batch_size, out_features]
        
        return output


class SAGELayer(nn.Module):
    """
    GraphSAGE 层
    实现采样和聚合操作
    """
    def __init__(self, in_features, out_features, aggregator='mean', num_samples=10, use_bias=False):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            aggregator: 聚合器类型 ('mean' 或 'maxpool')
            num_samples: 每个节点采样的邻居数量
            use_bias: 是否使用偏置
        """
        super(SAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_samples = num_samples
        self.aggregator_type = aggregator
        
        # 选择聚合器
        if aggregator == 'mean':
            self.aggregator = MeanAggregator(in_features, out_features, use_bias)
        elif aggregator == 'maxpool':
            self.aggregator = MaxPoolAggregator(in_features, out_features, use_bias)
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator}")

    def sample_neighbors(self, node_list, adj_list):
        """
        采样邻居节点
        
        Args:
            node_list: 节点列表 [batch_size]
            adj_list: 邻接列表（字典或列表的列表）
        Returns:
            sampled_neighbors: 采样的邻居 [batch_size, num_samples]
        """
        batch_size = len(node_list)
        sampled_neighbors = []
        
        for node in node_list:
            node = int(node.item() if torch.is_tensor(node) else node)
            if node in adj_list:
                neighbors = adj_list[node]
                # 如果邻居数量超过 num_samples，随机采样
                if len(neighbors) > self.num_samples:
                    sampled = random.sample(neighbors, self.num_samples)
                else:
                    # 如果邻居数量不足，重复采样
                    sampled = random.choices(neighbors, k=self.num_samples)
            else:
                # 如果没有邻居，使用自身节点
                sampled = [node] * self.num_samples
            sampled_neighbors.append(sampled)
        
        return sampled_neighbors

    def forward(self, features, node_list, adj_list):
        """
        Args:
            features: 所有节点的特征矩阵 [N, in_features]
            node_list: 当前层的节点列表 [batch_size]
            adj_list: 邻接列表（字典或列表的列表）
        Returns:
            输出特征 [batch_size, out_features]
        """
        # 获取自身特征
        node_indices = [int(node.item() if torch.is_tensor(node) else node) for node in node_list]
        self_features = features[node_indices]  # [batch_size, in_features]
        
        # 采样邻居
        sampled_neighbors = self.sample_neighbors(node_list, adj_list)
        
        # 聚合邻居特征
        if self.aggregator_type == 'mean':
            # 平均聚合：收集所有邻居特征并平均
            neigh_features_list = []
            for neighbors in sampled_neighbors:
                neigh_indices = [int(n) for n in neighbors]
                neigh_feat = features[neigh_indices]  # [num_samples, in_features]
                neigh_mean = torch.mean(neigh_feat, dim=0, keepdim=True)  # [1, in_features]
                neigh_features_list.append(neigh_mean)
            neigh_features = torch.cat(neigh_features_list, dim=0)  # [batch_size, in_features]
            
            output = self.aggregator(self_features, neigh_features)
            
        elif self.aggregator_type == 'maxpool':
            # 最大池化聚合：保留所有邻居特征用于池化
            neigh_features_list = []
            for neighbors in sampled_neighbors:
                neigh_indices = [int(n) for n in neighbors]
                neigh_feat = features[neigh_indices]  # [num_samples, in_features]
                neigh_features_list.append(neigh_feat.unsqueeze(0))  # [1, num_samples, in_features]
            neigh_features_batch = torch.cat(neigh_features_list, dim=0)  # [batch_size, num_samples, in_features]
            
            output = self.aggregator(self_features, neigh_features_batch)
        
        return output


class GraphSAGE(nn.Module):
    """
    图采样和聚合网络（Graph Sample and Aggregate）
    
    Paper: "Inductive Representation Learning on Large Graphs" (Hamilton et al., NIPS 2017)
    
    结构：
    - 第一层: SAGELayer(input_dim -> hidden_dim, ReLU激活, dropout=True)
    - 第二层: SAGELayer(hidden_dim -> output_dim, 无激活, dropout=True)
    """
    def __init__(self, nfeat, nhid, nclass, aggregator='mean', num_samples=10, dropout=0.5):
        """
        Args:
            nfeat: 输入特征维度
            nhid: 隐藏层维度
            nclass: 输出类别数
            aggregator: 聚合器类型 ('mean' 或 'maxpool')
            num_samples: 每层采样的邻居数量
            dropout: dropout 概率
        """
        super(GraphSAGE, self).__init__()
        self.dropout = dropout
        
        # 第一层：input_dim -> hidden_dim
        self.sage1 = SAGELayer(nfeat, nhid, aggregator=aggregator, num_samples=num_samples, use_bias=False)
        
        # 第二层：hidden_dim -> output_dim
        self.sage2 = SAGELayer(nhid, nclass, aggregator=aggregator, num_samples=num_samples, use_bias=False)

    def forward(self, features, node_list, adj_list):
        """
        Args:
            features: 所有节点的特征矩阵 [N, nfeat]
            node_list: 目标节点列表 [batch_size]
            adj_list: 邻接列表（字典或列表的列表）
        Returns:
            输出 logits [batch_size, nclass]（不包含 softmax）
        """
        # 如果输入是稀疏的，转换为密集矩阵
        if features.is_sparse:
            features = features.to_dense()
        
        # 第一层：SAGELayer + ReLU + Dropout
        # 对目标节点进行采样和聚合
        x = self.sage1(features, node_list, adj_list)  # [batch_size, nhid]
        x = F.relu(x)  # ReLU 激活
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 为了第二层能够访问第一层的输出，我们需要更新特征矩阵
        # 但为了效率，我们创建一个临时特征矩阵，只更新目标节点的特征
        node_indices = [int(node.item() if torch.is_tensor(node) else node) for node in node_list]
        
        # 创建更新后的特征矩阵（用于第二层）
        # 第一层的输出维度是 nhid，所以需要扩展特征矩阵
        features_updated = torch.zeros(features.size(0), self.sage1.out_features, 
                                      device=features.device, dtype=features.dtype)
        # 对于非目标节点，使用零向量（或者可以保持原始特征，但需要降维）
        # 这里简化处理：只更新目标节点的特征
        features_updated[node_indices] = x
        
        # 第二层：SAGELayer + Dropout（无激活函数）
        # 对目标节点再次进行采样和聚合（使用第一层的输出）
        x = self.sage2(features_updated, node_list, adj_list)  # [batch_size, nclass]
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x  # 返回 logits，不包含 softmax
    
    def predict(self, features, node_list, adj_list):
        """
        预测函数，返回 softmax 概率
        
        Args:
            features: 所有节点的特征矩阵 [N, nfeat]
            node_list: 目标节点列表 [batch_size]
            adj_list: 邻接列表
        Returns:
            softmax 概率 [batch_size, nclass]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(features, node_list, adj_list)
            return F.softmax(outputs, dim=1)

