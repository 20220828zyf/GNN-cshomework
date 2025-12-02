import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

# 兼容不同版本的 scipy
try:
    # 新版本 scipy (>= 1.8.0)
    from scipy.sparse.linalg import eigsh
except ImportError:
    # 旧版本 scipy (< 1.8.0)
    from scipy.sparse.linalg.eigen.arpack import eigsh


class GraphWaveletLayer(nn.Module):
    """
    图小波层（Graph Wavelet Layer）
    实现 GWNN 论文中的小波变换
    
    Paper: "Graph Wavelet Neural Network" (Xu et al., ICLR 2019)
    
    核心思想：
    - 使用图小波变换替代图卷积
    - 小波基函数：ψ_s = U g_s(Λ) U^T
    - 其中 U 是拉普拉斯矩阵的特征向量，g_s 是小波函数（如墨西哥帽小波）
    - 变换：X_wavelet = ψ_s X W
    - 逆变换：X' = ψ_s^T X_wavelet
    """
    def __init__(self, in_features, out_features, dropout=0.5, bias=False):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: dropout 概率
            bias: 是否使用偏置
        """
        super(GraphWaveletLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # 权重矩阵 W
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot 初始化（Xavier uniform）
        init_range = np.sqrt(6.0 / (self.in_features + self.out_features))
        nn.init.uniform_(self.weight, -init_range, init_range)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, wavelet_basis, inverse_wavelet_basis):
        """
        Args:
            input: 节点特征矩阵 [N, in_features]（可以是稀疏或密集张量）
            wavelet_basis: 图小波基矩阵 [N, N]（ψ_s）
            inverse_wavelet_basis: 逆小波基矩阵 [N, N]（ψ_s^T）
        Returns:
            输出特征矩阵 [N, out_features]
        """
        x = input
        
        # 如果输入是稀疏的，转换为密集矩阵
        if x.is_sparse:
            x = x.to_dense()
        
        # Dropout
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 小波变换：X_wavelet = ψ_s X
        # 先进行线性变换：XW
        xw = torch.mm(x, self.weight)  # [N, out_features]
        
        # 小波变换：ψ_s (XW)
        if wavelet_basis.is_sparse:
            x_wavelet = torch.spmm(wavelet_basis, xw)  # [N, out_features]
        else:
            x_wavelet = torch.mm(wavelet_basis, xw)  # [N, out_features]
        
        # 逆小波变换：ψ_s^T X_wavelet
        if inverse_wavelet_basis.is_sparse:
            output = torch.spmm(inverse_wavelet_basis, x_wavelet)  # [N, out_features]
        else:
            output = torch.mm(inverse_wavelet_basis, x_wavelet)  # [N, out_features]
        
        # 添加偏置（如果存在）
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GWNN(nn.Module):
    """
    图小波神经网络（Graph Wavelet Neural Network）
    
    结构：
    - 第一层: GraphWaveletLayer(input_dim -> hidden_dim, ReLU激活, dropout=True)
    - 第二层: GraphWaveletLayer(hidden_dim -> output_dim, 无激活, dropout=True)
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        """
        Args:
            nfeat: 输入特征维度
            nhid: 隐藏层维度
            nclass: 输出类别数
            dropout: dropout 概率
        """
        super(GWNN, self).__init__()
        
        # 第一层：input_dim -> hidden_dim，使用 ReLU 激活，有 dropout
        self.gw1 = GraphWaveletLayer(nfeat, nhid, dropout=dropout, bias=False)
        
        # 第二层：hidden_dim -> output_dim，无激活函数，有 dropout
        self.gw2 = GraphWaveletLayer(nhid, nclass, dropout=dropout, bias=False)
        
        self.dropout = dropout

    def forward(self, x, wavelet_basis, inverse_wavelet_basis):
        """
        Args:
            x: 节点特征矩阵 [N, nfeat]
            wavelet_basis: 图小波基矩阵 [N, N]（ψ_s）
            inverse_wavelet_basis: 逆小波基矩阵 [N, N]（ψ_s^T）
        Returns:
            输出 logits [N, nclass]（不包含 softmax）
        """
        # 第一层：GraphWaveletLayer + ReLU + Dropout
        x = self.gw1(x, wavelet_basis, inverse_wavelet_basis)
        x = F.relu(x)  # ReLU 激活
        
        # 第二层：GraphWaveletLayer + Dropout（无激活函数）
        x = self.gw2(x, wavelet_basis, inverse_wavelet_basis)
        
        return x  # 返回 logits，不包含 softmax
    
    def predict(self, x, wavelet_basis, inverse_wavelet_basis):
        """
        预测函数，返回 softmax 概率
        
        Args:
            x: 节点特征矩阵 [N, nfeat]
            wavelet_basis: 图小波基矩阵 [N, N]
            inverse_wavelet_basis: 逆小波基矩阵 [N, N]
        Returns:
            softmax 概率 [N, nclass]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, wavelet_basis, inverse_wavelet_basis)
            return F.softmax(outputs, dim=1)


def normalize_laplacian(adj):
    """
    归一化拉普拉斯矩阵：L = I - D^{-1/2} A D^{-1/2}
    
    Args:
        adj: 邻接矩阵（scipy sparse）
    Returns:
        归一化拉普拉斯矩阵（scipy sparse）
    """
    adj = sp.coo_matrix(adj)
    # 添加自环
    adj = adj + sp.eye(adj.shape[0])
    
    # 计算度矩阵的平方根倒数
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # 归一化：D^{-1/2} A D^{-1/2}
    adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    # 拉普拉斯矩阵：L = I - D^{-1/2} A D^{-1/2}
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    
    return laplacian


def compute_wavelet_basis(adj, s=1.0, method='heat'):
    """
    计算图小波基函数
    
    Args:
        adj: 邻接矩阵（scipy sparse）
        s: 小波尺度参数
        method: 小波函数类型 ('heat' 或 'mexican_hat')
            - 'heat': 热核小波 g_s(λ) = exp(-s*λ)
            - 'mexican_hat': 墨西哥帽小波 g_s(λ) = λ * exp(-s*λ)
    
    Returns:
        wavelet_basis: 小波基矩阵 [N, N]（ψ_s）
        inverse_wavelet_basis: 逆小波基矩阵 [N, N]（ψ_s^T）
    """
    # 计算归一化拉普拉斯矩阵
    laplacian = normalize_laplacian(adj)
    
    # 计算拉普拉斯矩阵的特征值和特征向量
    # 对于大图，只计算前 k 个特征值（k << N）
    N = adj.shape[0]
    k = min(200, N - 1)  # 限制特征值数量以提高效率
    
    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')
    except:
        # 如果特征分解失败，使用所有特征值
        laplacian_dense = laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_dense)
    
    # 确保特征值按升序排列
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 计算小波函数 g_s(λ)
    if method == 'heat':
        # 热核小波：g_s(λ) = exp(-s*λ)
        g_s = np.exp(-s * eigenvalues)
    elif method == 'mexican_hat':
        # 墨西哥帽小波：g_s(λ) = λ * exp(-s*λ)
        g_s = eigenvalues * np.exp(-s * eigenvalues)
    else:
        raise ValueError(f"Unknown wavelet method: {method}")
    
    # 构建对角矩阵 g_s(Λ)
    g_s_diag = sp.diags(g_s)
    
    # 计算小波基：ψ_s = U g_s(Λ) U^T
    # 由于我们只使用了部分特征向量，需要处理维度
    if eigenvectors.shape[1] < N:
        # 如果特征向量数量少于节点数，需要补零
        # 对于未计算的特征值，假设 g_s(λ) = 0
        full_g_s = np.zeros(N)
        full_g_s[:len(g_s)] = g_s
        g_s_diag = sp.diags(full_g_s)
        
        # 构建完整的特征向量矩阵（未计算的部分用零填充）
        full_eigenvectors = np.zeros((N, N))
        full_eigenvectors[:, :eigenvectors.shape[1]] = eigenvectors
        eigenvectors = full_eigenvectors
    
    # 计算 ψ_s = U g_s(Λ) U^T
    U_g_s = eigenvectors @ g_s_diag.toarray()
    wavelet_basis = U_g_s @ eigenvectors.T
    
    # 逆小波基：ψ_s^T = (U g_s(Λ) U^T)^T = U g_s(Λ) U^T（因为是对称的）
    # 实际上，对于对称矩阵，逆小波基就是小波基的转置
    inverse_wavelet_basis = wavelet_basis.T
    
    return wavelet_basis, inverse_wavelet_basis

