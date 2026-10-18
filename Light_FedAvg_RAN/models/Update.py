# models/Update.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# ---------------------- 辅助类：TabularDataset ----------------------
class TabularDataset(Dataset):
    """
    用于接收 NumPy 数组的 Dataset 类。
    在 main_fed.py 中，我们将利用此 Dataset 来封装从 sampling.py 中获取的客户端数据。
    """

    def __init__(self, X, y):
        # 确保输入是 PyTorch 张量
        if not torch.is_tensor(X):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = X.to(torch.float32)

        if not torch.is_tensor(y):
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = y.to(torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ---------------------- 核心类：LocalUpdate ----------------------

class LocalUpdate(object):
    """
    客户端本地模型的训练和更新类。
    """

    def __init__(self, args, client_data, test_data=None):
        """
        初始化本地更新环境。

        参数:
            args (object): 命令行参数对象，包含 local_ep, local_bs, lr 等。
            client_data (dict): 客户端的本地数据 (包含 'X' 和 'y' 的 NumPy 数组)。
            test_data (DataLoader, optional): 本地可选的测试数据加载器（通常不需要）。
        """
        self.args = args
        self.device = torch.device(args.device)
        self.loss_func = nn.CrossEntropyLoss()

        # 1. 创建本地训练 DataLoader
        local_dataset = TabularDataset(client_data['X'], client_data['y'])
        self.trainloader = DataLoader(
            local_dataset,
            batch_size=args.local_bs,
            shuffle=True,
            drop_last=False  # 避免在小客户端上丢失数据
        )
        self.testloader = test_data
        self.local_size = len(local_dataset)  # 记录本地数据总量

    def evaluate_local(self, model):
        """
        在本地训练集上，使用 model.eval() 模式进行精确评估。
        """
        model.eval()  # 切换到评估模式
        total_loss = 0
        correct_count = 0
        total_count = 0

        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                log_probs = model(images)

                # =========================================================
                # 【新增】针对 TabM 与传统模型 (RAN) 的不同评估逻辑
                # =========================================================
                if self.args.model_name == 'TabM':
                    # TabM 输出形状: (B, k, C)
                    B, K_models, C = log_probs.shape

                    # 计算 Loss：需要将 log_probs 和 labels 展平以匹配 k 个子模型的输出
                    log_probs_flat = log_probs.view(-1, C)  # 形状: (B*k, C)
                    labels_flat = labels.unsqueeze(1).expand(-1, K_models).reshape(-1)  # 形状: (B*k,)
                    loss = self.loss_func(log_probs_flat, labels_flat)

                    # 计算 Acc：对 k 个子模型的 Softmax 概率求平均 (集成软投票)
                    mean_probs = torch.softmax(log_probs, dim=-1).mean(dim=1)  # 形状: (B, C)
                    y_pred = mean_probs.argmax(dim=1, keepdim=True)

                else:
                    # 传统模型 (RAN) 输出形状: (B, C)
                    loss = self.loss_func(log_probs, labels)
                    y_pred = log_probs.data.max(1, keepdim=True)[1]
                # =========================================================

                total_loss += loss.item() * labels.size(0)
                correct_count += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                total_count += labels.size(0)

        # 返回整个本地数据集的平均损失和准确率
        return total_loss / total_count, correct_count.item() / total_count

    def train(self, model, global_round=0):
        """
        在客户端本地数据上训练模型 local_ep 轮。
        """
        # 1. 切换到训练模式并设置优化器
        model.to(self.device)
        model.train()

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        # 2. 执行本地训练 local_ep 轮
        for iter_ in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                log_probs = model(images)

                # =========================================================
                # 【新增】针对 TabM 与传统模型 (RAN) 的不同 Loss 计算逻辑
                # =========================================================
                if self.args.model_name == 'TabM':
                    # TabM: 并行计算 k 个隐式子模型的交叉熵损失
                    B, K_models, C = log_probs.shape
                    log_probs_flat = log_probs.view(-1, C)
                    labels_flat = labels.unsqueeze(1).expand(-1, K_models).reshape(-1)
                    loss = self.loss_func(log_probs_flat, labels_flat)
                else:
                    # 传统模型 (RAN): 直接计算交叉熵损失
                    loss = self.loss_func(log_probs, labels)
                # =========================================================

                # 反向传播和优化
                loss.backward()
                optimizer.step()

        # 3. 本地评估并返回结果
        final_loss, final_acc = self.evaluate_local(model)

        # 确保模型回到 CPU 内存，并获取更新后的权重字典
        model.to('cpu')

        return model.state_dict(), final_loss, final_acc