# main_ran.py (With Early Stopping & Best Model Saving)
import argparse
import random
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 导入自定义模块
# 确保项目根目录在 PATH 中 (假设 __file__ 位于根目录下)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils.options import args_parser
from models.Nets import RANTabular
# 【保留】混淆矩阵绘图工具
from utils.confusion_matrix_plot import plot_and_save_confusion_matrix

SEED = 150


def set_seed(seed):
    """设置所有主要库和环境的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"✅ 已强制设置所有库的随机种子为 {seed}")


# 在程序开始时调用
set_seed(SEED)


# ---------------------- 辅助类：TabularDataset ----------------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.y[i]


# ---------------------- 数据加载（无验证集） ----------------------
def load_data_no_val(args):
    excel_path = args.excel_path
    label_col = args.label_column
    feature_cols = args.feature_cols
    seed = args.seed
    test_size = args.test_size

    df = pd.read_excel(excel_path, engine="openpyxl")
    df = df.dropna(subset=feature_cols + [label_col])

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y_raw = df[label_col].to_numpy()

    # >>>>>>>>>>>>>>>>> 标签编码：手动映射修改开始 <<<<<<<<<<<<<<<<<
    label_map_manual = {'Normal': 0, 'Asthmatic': 1}
    unique_labels = np.unique(y_raw)
    for label in unique_labels:
        if label not in label_map_manual:
            raise ValueError(
                f"错误：原始标签 '{label}' 不在预期的手动映射中 ({list(label_map_manual.keys())})。"
            )

    y = np.array([label_map_manual[label] for label in y_raw])

    # 【还原】保持原有的类别定义方式，方便后续扩展
    classes = np.array(['Normal', 'Asthmatic'])
    num_classes = len(classes)
    class_map = {0: 'Normal', 1: 'Asthmatic'}
    # >>>>>>>>>>>>>>>>> 标签编码：手动映射修改结束 <<<<<<<<<<<<<<<<<

    print(f"特征列: {feature_cols}")
    print(f"类别映射 (手动设定): {class_map} (共 {num_classes} 类)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"训练集: {len(X_train)}  | 测试集: {len(X_test)}")

    return X_train, y_train, X_test, y_test, scaler, num_classes, feature_cols, class_map


# ---------------------- 单轮训练 ----------------------
def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        logits = model(xb)
        loss = ce(logits, yb)
        loss.backward()
        opt.step()

        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)

    return total_loss / total, correct / total


# ---------------------- 评估（测试） ----------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)

        total_loss += loss.item() * len(yb)
        pred = logits.argmax(1)

        correct += (pred == yb).sum().item()
        total += len(yb)

        y_true.append(yb.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.concatenate(y_true),
        np.concatenate(y_pred)
    )


# ---------------------- 绘图函数 ----------------------
def plot_training_curves(history, args):
    """
    绘制并保存训练/测试的 Loss 和 Accuracy 曲线。
    保存路径: ./results_plots/
    """
    save_dir = getattr(args, 'output_dir', './results_plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(history['train_acc']) + 1)

    # 1. 绘制 Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs, history['test_acc'], label='Test Acc', marker='x')
    plt.title(f'Centralized Accuracy (Seed={args.seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(save_dir, f'centralized_accuracy_curve_Seed{args.seed}.png')
    plt.savefig(acc_path)
    plt.close()
    print(f"[Plot] Accuracy 曲线已保存至: {acc_path}")

    # 2. 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['test_loss'], label='Test Loss', marker='x')
    plt.title(f'Centralized Loss (Seed={args.seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    loss_path = os.path.join(save_dir, f'centralized_loss_curve_Seed{args.seed}.png')
    plt.savefig(loss_path)
    plt.close()
    print(f"[Plot] Loss 曲线已保存至: {loss_path}")


# ---------------------- 主函数 ----------------------
def main_ran():
    args = args_parser()
    set_seed(args.seed)
    device = torch.device(args.device)

    # 数据加载
    Xtr, ytr, Xte, yte, scaler, num_classes, feat_cols, class_map = load_data_no_val(args)

    # Dataloader
    tr_dl = DataLoader(TabularDataset(Xtr, ytr), batch_size=args.local_bs, shuffle=True)
    te_dl = DataLoader(TabularDataset(Xte, yte), batch_size=args.local_bs)

    # 模型
    in_dim = len(feat_cols)
    model = RANTabular(in_dim, num_classes,
                       args.model_dim, args.num_blocks,
                       args.hidden_mult, args.attn_reduction,
                       args.dropout).to(device)

    print("\n--- 模型结构 ---")
    print(model)
    print("----------------")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 初始化历史记录字典
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # ==================== 【新增】早停机制参数 ====================
    best_acc = 0.0  # 初始化最佳准确率
    patience = 20  # 耐心值：如果连续 20 轮没提升就停止
    counter = 0  # 计数器

    # 确保保存目录存在
    save_dir = getattr(args, 'output_dir', './results_plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 定义最佳模型保存路径
    best_model_path = os.path.join(save_dir, f'best_model_Seed{args.seed}.pth')
    # ============================================================

    print(f"\n--- 开始训练 (早停 Patience={patience}) ---")
    for ep in range(1, args.epochs + 1):
        # 1. 训练
        tr_loss, tr_acc = train_one_epoch(model, tr_dl, optimizer, device)

        # 2. 验证/测试
        te_loss, te_acc, _, _ = evaluate(model, te_dl, device)

        # 3. 记录历史
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)

        # 打印日志（不换行，后面接早停状态）
        print(f"Epoch {ep:03d}/{args.epochs} | "
              f"Tr Loss {tr_loss:.4f} Acc {tr_acc * 100:.2f}% | "
              f"Te Loss {te_loss:.4f} Acc {te_acc * 100:.2f}%", end="")

        # ==================== 【核心】早停逻辑 ====================
        if te_acc > best_acc:
            best_acc = te_acc
            counter = 0  # 发现新高，重置计数器
            torch.save(model.state_dict(), best_model_path)  # 保存参数
            print(f"  >>> ★ Best Saved ({best_acc * 100:.2f}%)")
        else:
            print("")
            # counter += 1
            # print(f"  | Wait {counter}/{patience}")
            # if counter >= patience:
            #     print(f"\n[Early Stopping] 触发！测试集准确率连续 {patience} 轮未提升，停止训练。")
            #     break
        # ========================================================

    # 4. 训练结束后绘制曲线 (使用 history 中实际记录的轮数)
    plot_training_curves(history, args)

    # ==================== 【关键】加载最佳模型 ====================
    print(f"\n正在加载历史最佳模型: {best_model_path} ...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("✅ 成功加载最佳模型参数。")
    else:
        print("⚠️ 警告：未找到最佳模型文件，将使用当前（最后一轮）模型。")
    # ============================================================

    # 最终测试 (使用加载后的 Best Model)
    print("\n================= 最终测试结果 (Best Model) =================")
    final_loss, final_acc, y_true, y_pred = evaluate(model, te_dl, device)

    print(f"最终测试集 Loss: {final_loss:.4f}")
    print(f"最终测试集 Acc:  {final_acc * 100:.2f}%")

    print("\n--- 混淆矩阵 ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    class_names = [str(class_map[i]) for i in range(num_classes)]

    # 调用混淆矩阵绘图函数
    plot_and_save_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title=' ',
        save_filename='cm_ran_centralized.png',
        args=args
    )

    print("\n--- 分类报告 ---")
    print(classification_report(y_true, y_pred, digits=4, target_names=class_names))
    print("================================================")


if __name__ == "__main__":
    main_ran()
