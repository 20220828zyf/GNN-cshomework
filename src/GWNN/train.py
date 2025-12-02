import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from gwnn_model import GWNN, compute_wavelet_basis
from utils import load_data_pytorch, sparse_mx_to_torch_sparse_tensor


def accuracy(output, labels):
    """计算准确率"""
    preds = output.max(1)[1]
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def train(dataset='cora', learning_rate=0.01, epochs=200, nhid=16, 
          dropout=0.5, weight_decay=5e-4, early_stopping=10, 
          wavelet_s=1.0, wavelet_method='heat', data_dir=None):
    """
    训练 GWNN 模型
    
    Args:
        dataset: 数据集名称 ('cora', 'citeseer', 'pubmed')
        learning_rate: 学习率
        epochs: 训练轮数
        nhid: 隐藏层维度
        dropout: dropout 概率
        weight_decay: 权重衰减（L2 正则化）
        early_stopping: early stopping 的 patience
        wavelet_s: 小波尺度参数
        wavelet_method: 小波函数类型 ('heat' 或 'mexican_hat')
        data_dir: 数据目录路径（如果为 None，则使用默认路径：项目根目录下的 src/data）
    """
    # 加载数据（已转换为 PyTorch 格式）
    adj, features, labels, idx_train, idx_val, idx_test = load_data_pytorch(
        dataset, data_dir=data_dir
    )

    # 转为 torch.tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 在移到设备之前，先计算类别数和验证标签
    unique_labels = torch.unique(labels)
    nclass = len(unique_labels)
    
    # 验证标签范围
    if labels.min().item() < 0 or labels.max().item() >= nclass:
        print(f"Warning: Label values out of range [0, {nclass-1}]")
        print(f"  Min label: {labels.min().item()}, Max label: {labels.max().item()}")
        print(f"  Unique labels: {unique_labels.tolist()}")
        # 重新映射标签到连续范围 [0, nclass-1]
        label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
        labels = torch.tensor([label_map[label.item()] for label in labels], dtype=torch.long)
        print(f"  Labels remapped to range [0, {nclass-1}]")

    # 计算图小波基函数
    print(f"Computing graph wavelet basis (method={wavelet_method}, s={wavelet_s})...")
    wavelet_basis, inverse_wavelet_basis = compute_wavelet_basis(
        adj, s=wavelet_s, method=wavelet_method
    )
    
    # 转换为 PyTorch 稀疏张量
    wavelet_basis = sparse_mx_to_torch_sparse_tensor(wavelet_basis)
    inverse_wavelet_basis = sparse_mx_to_torch_sparse_tensor(inverse_wavelet_basis)

    # 将数据移到设备上
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    wavelet_basis = wavelet_basis.to(device)
    inverse_wavelet_basis = inverse_wavelet_basis.to(device)

    # 定义模型
    model = GWNN(
        nfeat=features.shape[1],
        nhid=nhid,
        nclass=nclass,
        dropout=dropout
    ).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Model: GWNN")
    print(f"Dataset: {dataset}")
    print(f"Wavelet method: {wavelet_method}, scale: {wavelet_s}")
    print(f"Training on {device}")

    # ------- training -------
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, wavelet_basis, inverse_wavelet_basis)  # 输出 logits

        # 计算损失
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        # 验证集
        model.eval()
        with torch.no_grad():
            output = model(features, wavelet_basis, inverse_wavelet_basis)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

        # Early stopping
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {loss_train.item():.5f} Acc: {acc_train:.5f} | "
                  f"Val Loss: {loss_val.item():.5f} Acc: {acc_val:.5f}")
        
        if patience_counter >= early_stopping:
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # ------- test -------
    model.eval()
    with torch.no_grad():
        output = model(features, wavelet_basis, inverse_wavelet_basis)
        # 确保所有索引都在有效范围内
        max_idx = min(len(output), len(labels))
        valid_mask = (idx_test < max_idx) & (idx_test >= 0)
        idx_test_valid = idx_test[valid_mask]
        
        if len(idx_test_valid) < len(idx_test):
            print(f"Warning: {len(idx_test) - len(idx_test_valid)} test indices out of range [0, {max_idx-1}]")
            print(f"  Using {len(idx_test_valid)} valid indices out of {len(idx_test)} total")
        
        if len(idx_test_valid) > 0:
            # 确保标签值在有效范围内
            test_labels = labels[idx_test_valid]
            if test_labels.max().item() >= nclass or test_labels.min().item() < 0:
                print(f"Error: Test labels out of range [0, {nclass-1}]")
                print(f"  Min: {test_labels.min().item()}, Max: {test_labels.max().item()}")
            else:
                loss_test = F.cross_entropy(output[idx_test_valid], test_labels)
                acc_test = accuracy(output[idx_test_valid], test_labels)
                print(f"Test set results: cost= {loss_test.item():.5f}, accuracy= {acc_test:.5f}")
        else:
            print("Error: No valid test indices found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GWNN model')
    parser.add_argument('--dataset', type=str, default='cora', 
                        choices=['cora', 'citeseer', 'pubmed'],
                        help='Dataset name (default: cora)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--nhid', type=int, default=16,
                        help='Number of hidden units (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Tolerance for early stopping (default: 10)')
    parser.add_argument('--wavelet_s', type=float, default=1.0,
                        help='Wavelet scale parameter (default: 1.0)')
    parser.add_argument('--wavelet_method', type=str, default='heat',
                        choices=['heat', 'mexican_hat'],
                        help='Wavelet function type (default: heat)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory path (default: None, uses src/data relative to project root)')
    
    args = parser.parse_args()
    
    train(
        dataset=args.dataset,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        nhid=args.nhid,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        wavelet_s=args.wavelet_s,
        wavelet_method=args.wavelet_method,
        data_dir=args.data_dir
    )

