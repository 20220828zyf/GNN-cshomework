import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import numpy as np
from graphsage_model import GraphSAGE
from utils import load_data_pytorch

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def accuracy(output, labels):
    """计算准确率"""
    preds = output.max(1)[1]
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def train(dataset='cora', learning_rate=0.01, epochs=200, nhid=16, 
          dropout=0.5, weight_decay=5e-4, early_stopping=10, 
          aggregator='mean', num_samples=10, batch_size=512, data_dir=None):
    """
    训练 GraphSAGE 模型
    
    Args:
        dataset: 数据集名称 ('cora', 'citeseer', 'pubmed')
        learning_rate: 学习率
        epochs: 训练轮数
        nhid: 隐藏层维度
        dropout: dropout 概率
        weight_decay: 权重衰减（L2 正则化）
        early_stopping: early stopping 的 patience
        aggregator: 聚合器类型 ('mean' 或 'maxpool')
        num_samples: 每层采样的邻居数量
        batch_size: 批处理大小
        data_dir: 数据目录路径（如果为 None，则使用默认路径：项目根目录下的 src/data）
    """
    # 设置随机种子
    set_seed(42)
    
    # 加载数据（已转换为 PyTorch 格式）
    features, labels, idx_train, idx_val, idx_test, adj_list = load_data_pytorch(
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

    # 将数据移到设备上
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    # 定义模型
    model = GraphSAGE(
        nfeat=features.shape[1],
        nhid=nhid,
        nclass=nclass,
        aggregator=aggregator,
        num_samples=num_samples,
        dropout=dropout
    ).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Model: GraphSAGE")
    print(f"Dataset: {dataset}")
    print(f"Aggregator: {aggregator}, Num samples: {num_samples}")
    print(f"Training on {device}")

    # ------- training -------
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        
        # 训练：使用批处理
        train_loss = 0
        train_acc = 0
        num_batches = 0
        
        # 将训练集分成批次
        train_indices = idx_train.cpu().numpy()
        np.random.shuffle(train_indices)
        
        for i in range(0, len(train_indices), batch_size):
            batch_nodes = train_indices[i:i+batch_size]
            batch_nodes_tensor = torch.LongTensor(batch_nodes).to(device)
            batch_labels = labels[batch_nodes_tensor]
            
            optimizer.zero_grad()
            output = model(features, batch_nodes_tensor, adj_list)  # 输出 logits
            
            # 计算损失
            loss = F.cross_entropy(output, batch_labels)
            acc = accuracy(output, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()
            num_batches += 1
        
        train_loss /= num_batches
        train_acc /= num_batches

        # 验证集
        model.eval()
        with torch.no_grad():
            # 验证集也使用批处理
            val_loss = 0
            val_acc = 0
            num_val_batches = 0
            
            val_indices = idx_val.cpu().numpy()
            for i in range(0, len(val_indices), batch_size):
                batch_nodes = val_indices[i:i+batch_size]
                batch_nodes_tensor = torch.LongTensor(batch_nodes).to(device)
                batch_labels = labels[batch_nodes_tensor]
                
                output = model(features, batch_nodes_tensor, adj_list)
                loss = F.cross_entropy(output, batch_labels)
                acc = accuracy(output, batch_labels)
                
                val_loss += loss.item()
                val_acc += acc.item()
                num_val_batches += 1
            
            val_loss /= num_val_batches
            val_acc /= num_val_batches

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.5f} Acc: {train_acc:.5f} | "
                  f"Val Loss: {val_loss:.5f} Acc: {val_acc:.5f}")
        
        if patience_counter >= early_stopping:
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # ------- test -------
    model.eval()
    with torch.no_grad():
        # 测试集也使用批处理
        test_loss = 0
        test_acc = 0
        num_test_batches = 0
        
        test_indices = idx_test.cpu().numpy()
        for i in range(0, len(test_indices), batch_size):
            batch_nodes = test_indices[i:i+batch_size]
            batch_nodes_tensor = torch.LongTensor(batch_nodes).to(device)
            batch_labels = labels[batch_nodes_tensor]
            
            output = model(features, batch_nodes_tensor, adj_list)
            loss = F.cross_entropy(output, batch_labels)
            acc = accuracy(output, batch_labels)
            
            test_loss += loss.item()
            test_acc += acc.item()
            num_test_batches += 1
        
        test_loss /= num_test_batches
        test_acc /= num_test_batches
        
        print(f"Test set results: cost= {test_loss:.5f}, accuracy= {test_acc:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GraphSAGE model')
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
    parser.add_argument('--aggregator', type=str, default='mean',
                        choices=['mean', 'maxpool'],
                        help='Aggregator type (default: mean)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of neighbors to sample per layer (default: 10)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training (default: 512)')
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
        aggregator=args.aggregator,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )

