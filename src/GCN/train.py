import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from gcn_model import GCN
from utils import load_data_pytorch


def train(model='gcn', dataset='cora', learning_rate=0.01, epochs=500, hidden1=16, 
          dropout=0.5, weight_decay=5e-4, early_stopping=20, max_degree=3, data_dir=None):
    """
    训练 GCN 模型
    对应原始 TensorFlow 实现中的训练流程
    
    Args:
        model: 模型类型 ('gcn' 或 'gcn_cheby')
        dataset: 数据集名称 ('cora', 'citeseer', 'pubmed')
        learning_rate: 学习率
        epochs: 训练轮数
        hidden1: 隐藏层维度
        dropout: dropout 概率
        weight_decay: 权重衰减（L2 正则化）
        early_stopping: early stopping 的 patience
        max_degree: Chebyshev 多项式的最大阶数（仅当 model='gcn_cheby' 时使用）
        data_dir: 数据目录路径（如果为 None，则使用默认路径：项目根目录下的 src/data）
    """
    # 加载数据（已转换为 PyTorch 格式）
    support, features, labels, idx_train, idx_val, idx_test = load_data_pytorch(
        dataset, data_dir=data_dir, model=model, max_degree=max_degree
    )

    # 转为 torch.tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 在移到设备之前，先计算类别数和验证标签
    # 计算类别数：使用唯一标签的数量
    # 这比使用 max() + 1 更安全，因为标签可能不连续
    unique_labels = torch.unique(labels)
    nclass = len(unique_labels)
    
    # 验证标签范围：确保所有标签值都在 [0, nclass-1] 范围内
    if labels.min().item() < 0 or labels.max().item() >= nclass:
        print(f"Warning: Label values out of range [0, {nclass-1}]")
        print(f"  Min label: {labels.min().item()}, Max label: {labels.max().item()}")
        print(f"  Unique labels: {unique_labels.tolist()}")
        # 重新映射标签到连续范围 [0, nclass-1]
        label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
        labels = torch.tensor([label_map[label.item()] for label in labels], dtype=torch.long)
        print(f"  Labels remapped to range [0, {nclass-1}]")

    # 将 support 列表中的所有矩阵移到设备上
    support = [s.to(device) for s in support]
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    # 定义模型（对应原始实现中的两层 GCN）
    num_supports = len(support)
    
    model_gcn = GCN(
        nfeat=features.shape[1],
        nhid=hidden1,
        nclass=nclass,
        num_supports=num_supports,
        dropout=dropout
    ).to(device)

    # 优化器（对应原始实现：Adam, lr=learning_rate, weight_decay=weight_decay）
    optimizer = optim.Adam(model_gcn.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Model: {model.upper()}")
    print(f"Dataset: {dataset}")
    print(f"Number of supports: {num_supports}")
    print(f"Training on {device}")

    # ------- training -------
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model_gcn.train()
        optimizer.zero_grad()
        output = model_gcn(features, support)  # 输出 logits

        # 计算损失（对应原始实现中的 masked_softmax_cross_entropy）
        # 由于输出是 logits，使用 cross_entropy（包含 softmax + nll_loss）
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # 添加权重衰减损失（对应原始实现中的 weight_decay * L2_loss）
        # 注意：PyTorch 的 weight_decay 参数已经处理了 L2 正则化

        loss_train.backward()
        optimizer.step()

        # 验证集
        model_gcn.eval()
        with torch.no_grad():
            output = model_gcn(features, support)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

        # Early stopping（对应原始实现）
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
    model_gcn.eval()
    with torch.no_grad():
        output = model_gcn(features, support)
        # 确保所有索引都在有效范围内
        max_idx = min(len(output), len(labels))
        # 过滤出有效的测试索引
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


def accuracy(output, labels):
    preds = output.max(1)[1]
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GCN model')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gcn_cheby'],
                        help='Model type: gcn or gcn_cheby (default: gcn)')
    parser.add_argument('--dataset', type=str, default='cora', 
                        choices=['cora', 'citeseer', 'pubmed'],
                        help='Dataset name (default: cora)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--hidden1', type=int, default=16,
                        help='Number of units in hidden layer 1 (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Tolerance for early stopping (default: 10)')
    parser.add_argument('--max_degree', type=int, default=3,
                        help='Maximum Chebyshev polynomial degree (default: 3)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory path (default: None, uses src/data relative to project root)')
    
    args = parser.parse_args()
    
    train(
        model=args.model,
        dataset=args.dataset,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hidden1=args.hidden1,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        max_degree=args.max_degree,
        data_dir=args.data_dir
    )
