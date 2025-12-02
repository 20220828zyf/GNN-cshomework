import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import sys
import os

# 获取项目根目录（假设 utils.py 在 src/GAT/ 目录下）
# 从 src/GAT/utils.py 向上两级到项目根目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str, data_dir=None):
    """
    加载数据
    
    Args:
        dataset_str: 数据集名称（如 'cora', 'citeseer', 'pubmed'）
        data_dir: 数据目录路径（如果为 None，则使用默认路径：项目根目录下的 src/data）
    
    Returns:
        adj: 邻接矩阵（scipy sparse）
        features: 特征矩阵（scipy sparse）
        labels: 标签（numpy array，one-hot）
        idx_train: 训练集索引
        idx_val: 验证集索引
        idx_test: 测试集索引
    """
    # 如果没有指定数据目录，使用默认路径（相对于项目根目录）
    if data_dir is None:
        data_dir = os.path.join(_PROJECT_ROOT, 'src', 'data')
    # 如果提供的是相对路径，也基于项目根目录解析
    elif not os.path.isabs(data_dir):
        data_dir = os.path.join(_PROJECT_ROOT, data_dir)
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    
    for i in range(len(names)):
        file_path = os.path.join(data_dir, f"ind.{dataset_str}.{names[i]}")
        with open(file_path, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_file = os.path.join(data_dir, f"ind.{dataset_str}.test.index")
    test_idx_reorder = parse_index_file(test_idx_file)
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # 转换为 PyTorch 格式
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    # 将标签转换为类别索引（而不是 one-hot）
    # 使用 argmax 来获取每个样本的类别索引
    labels = torch.LongTensor(np.argmax(labels, axis=1))
    
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(adj):
    """对称归一化邻接矩阵"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """
    预处理邻接矩阵
    返回归一化的邻接矩阵：D^{-1/2} (A + I) D^{-1/2}
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def preprocess_features(features):
    """
    预处理特征矩阵
    行归一化特征矩阵
    """
    rowsum = np.array(features.sum(1))
    # 避免除以零：将零行和设置为 1
    rowsum[rowsum == 0] = 1.0
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将 scipy 稀疏矩阵转换为 PyTorch 稀疏张量
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # 使用新的 API 替代已弃用的 torch.sparse.FloatTensor
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def load_data_pytorch(dataset_str, data_dir=None):
    """
    加载数据并转换为 PyTorch 格式的完整函数
    
    Args:
        dataset_str: 数据集名称
        data_dir: 数据目录路径（如果为 None，则使用默认路径：项目根目录下的 src/data）
    
    Returns:
        adj: 邻接矩阵（PyTorch 稀疏张量，已添加自环）
        features: 特征矩阵（PyTorch 稀疏或密集张量）
        labels: 标签（PyTorch LongTensor，类别索引）
        idx_train: 训练集索引（PyTorch LongTensor）
        idx_val: 验证集索引（PyTorch LongTensor）
        idx_test: 测试集索引（PyTorch LongTensor）
    """
    # 加载原始数据
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset_str, data_dir)
    
    # 预处理邻接矩阵：添加自环并转换为 PyTorch 格式
    # GAT 需要自环，因为每个节点需要关注自己
    adj = adj + sp.eye(adj.shape[0])  # 添加自环
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    # 预处理特征矩阵
    features = preprocess_features(features)
    features = sparse_mx_to_torch_sparse_tensor(features)
    
    return adj, features, labels, idx_train, idx_val, idx_test

