# models/Update_light_mix.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# ---------------------- 辅助类：TabularDataset ----------------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
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


# ---------------------- 核心稀疏化函数 (含文献1误差补偿) ----------------------

def apply_topk_sparsity(delta_weights, sparsity, old_residuals=None):
    """
    对模型权重差分 (Delta W) 应用 Top-K 稀疏化。
    【修改】已集成 Aji & Heafield (2017) 的误差补偿 (Residual) 逻辑。

    参数:
        delta_weights (dict): 本轮计算出的权重差分。
        sparsity (float): 稀疏化率。
        old_residuals (dict, optional): 上一轮未发送的梯度残差。

    返回:
        (sparse_delta_w, new_residuals): 稀疏化后的权重 和 本轮产生的残差。
    """
    if sparsity == 0.0:
        return delta_weights, {}  # 不稀疏化，也不产生残差

    sparse_delta_w = {}
    new_residuals = {}

    # 收集所有浮点参数的绝对值 (基于 累积梯度)
    all_abs_values = []
    keys_to_process = []
    accumulated_delta = {}  # 暂存: Current Delta + Old Residual

    for key, tensor in delta_weights.items():
        # 排除非浮点型参数 (如 num_batches_tracked)
        if tensor.dtype.is_floating_point:
            # --- 误差补偿核心逻辑 ---
            if old_residuals and key in old_residuals:
                accumulated = tensor + old_residuals[key]
            else:
                accumulated = tensor.clone()

            accumulated_delta[key] = accumulated
            all_abs_values.append(accumulated.abs().flatten())
            keys_to_process.append(key)
        else:
            # 非浮点型参数直接保留原始值，不参与稀疏化
            sparse_delta_w[key] = tensor

    if not all_abs_values:
        return sparse_delta_w, {}

    # 1. 计算全局 Top-K 阈值
    all_abs_values = torch.cat(all_abs_values)
    num_params = len(all_abs_values)
    k = max(1, int(num_params * (1.0 - sparsity)))

    threshold = all_abs_values.kthvalue(num_params - k + 1).values.item()

    # 2. 应用阈值稀疏化 并 更新残差
    for key in keys_to_process:
        tensor_acc = accumulated_delta[key]

        # 生成掩码
        mask = tensor_acc.abs() >= threshold

        # 稀疏张量 = 累积值 * 掩码 (被选中的发送)
        sparse_tensor = tensor_acc * mask.to(tensor_acc.dtype)
        sparse_delta_w[key] = sparse_tensor

        # 新残差 = 累积值 - 被发送的部分 (没被选中的留下)
        new_residuals[key] = tensor_acc - sparse_tensor

    return sparse_delta_w, new_residuals


# ---------------------- 【新增】部署优化：基于标准差的非结构化剪枝 ----------------------

def apply_songhan_pruning(model, qp_conv=0.5, qp_fc=1.0):
    """
    【新增】对训练好的全局模型执行基于逐层标准差的非结构化剪枝 (Song Han 方法)。

    参数:
        model: 训练完毕的全局模型
        qp_conv: 卷积层的质量参数 (敏感层，值较小)
        qp_fc: 全连接层的质量参数 (冗余层，值较大)
    返回:
        masks (dict): 记录每一层哪些权重被保留 (1) 哪些被剪枝 (0)
    """
    masks = {}
    model.to('cpu')

    with torch.no_grad():
        for name, module in model.named_modules():
            # 判断层类型，分别处理
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                weight = module.weight.data
                # 核心逻辑：阈值 = 质量参数 * 标准差
                threshold = qp_conv * torch.std(weight)
                mask = torch.abs(weight) >= threshold
                # 将低于阈值的权重清零
                module.weight.data.mul_(mask.float())
                masks[name + '.weight'] = mask.float()

            elif isinstance(module, nn.Linear):
                weight = module.weight.data
                threshold = qp_fc * torch.std(weight)
                mask = torch.abs(weight) >= threshold
                module.weight.data.mul_(mask.float())
                masks[name + '.weight'] = mask.float()

    return masks


# ---------------------- 核心类：LocalUpdate_LightMix ----------------------

class LocalUpdate_LightMix(object):
    """
    客户端本地模型的训练和更新类，具有：
    1. 冻结层 (本地加速) - 【保留】
    2. 稀疏 Delta-weight 上传 (通信优化) - 【已恢复残差补偿】
    3. 剪枝后微调梯度冻结 (部署优化) - 【新增】
    """

    def __init__(self, args, client_data, test_data=None):
        self.args = args
        self.device = torch.device(args.device)
        self.loss_func = nn.CrossEntropyLoss()

        local_dataset = TabularDataset(client_data['X'], client_data['y'])
        self.trainloader = DataLoader(
            local_dataset,
            batch_size=args.local_bs,
            shuffle=True,
            drop_last=False
        )
        self.testloader = test_data
        self.local_size = len(local_dataset)

    def evaluate_local(self, model):
        model.eval()
        total_loss = 0
        correct_count = 0
        total_count = 0

        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                log_probs = model(images)
                loss = self.loss_func(log_probs, labels)
                total_loss += loss.item() * labels.size(0)
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct_count += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                total_count += labels.size(0)

        return total_loss / total_count, correct_count.item() / total_count

    # 【修改】新增 masks 参数，用于微调阶段传入全局掩码
    def train(self, model, global_round=0, w_global=None, client_idx=None, old_residual=None, masks=None):
        """
        本地训练，实现轻量化策略。
        """
        c_id = client_idx if client_idx is not None else '?'

        # 1. 切换到训练模式并设置优化器
        model.to(self.device)
        model.train()

        # --- 策略 I: 冻结层 (本地加速) --- 【保留】
        # (注：此处通过设置 requires_grad=False 冻结指定数量的残差块参数，加速本地计算)
        if self.args.freeze_blocks_count > 0:
            freeze_count = self.args.freeze_blocks_count
            actual_freeze = min(freeze_count, len(model.blocks))
            for i in range(actual_freeze):
                block = model.blocks[i]
                for param in block.parameters():
                    param.requires_grad = False

        # 只对非冻结层进行优化
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        # 2. 执行本地训练 local_ep 轮
        for iter_ in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                # --- 【新增】策略 III: 训练后非结构化剪枝微调核心逻辑 ---
                # 在梯度更新之前，用掩码将已剪枝位置的梯度置零，防止权重死灰复燃
                if masks is not None:
                    for name, param in model.named_parameters():
                        if name in masks and param.grad is not None:
                            # 掩码中 0 的位置，梯度变成 0，从而该位置的权重不会被更新
                            param.grad.data.mul_(masks[name].to(self.device))

                optimizer.step()

        # 3. 本地评估并获取本地权重
        final_loss, final_acc = self.evaluate_local(model)
        model.to('cpu')
        w_local = model.state_dict()

        # 4. --- 策略 II: 结构稀疏 Delta-weight (通信优化) ---
        if w_global is None:
            raise ValueError("w_global 必须传入 LocalUpdate_LightMix.train 以计算 Delta W。")

        # 计算 Delta W
        delta_w = {}
        for key in w_local.keys():
            if w_local[key].dtype.is_floating_point:
                delta_w[key] = w_local[key].data.clone()
                # 本地训练结束后 model 已移至 CPU；全局权重可能仍位于 CUDA。
                # 计算差值前将对应全局张量对齐到本地权重所在设备。
                global_tensor = w_global[key].data.to(delta_w[key].device)
                delta_w[key].sub_(global_tensor)  # w_local - w_global
            else:
                delta_w[key] = w_local[key].data.clone()

        # 应用 Top-K 稀疏化 (带误差补偿)
        if self.args.delta_weight_method == 'sparse':

            # 传入 old_residual，接收 new_residual
            sparse_delta_w, new_residual = apply_topk_sparsity(
                delta_w,
                self.args.delta_sparsity,
                old_residuals=old_residual
            )

            # 统计稀疏化效果 (仅用于打印)
            non_zero_count = sum(torch.count_nonzero(t) for k, t in sparse_delta_w.items() if t.dtype.is_floating_point)
            total_count = sum(t.numel() for k, t in sparse_delta_w.items() if t.dtype.is_floating_point)
            effective_sparsity = 1.0 - (non_zero_count.item() / total_count) if total_count > 0 else 0.0

            print(f"  > C{c_id} | Loss: {final_loss:.4f} Acc: {final_acc * 100:.2f}% | "
                  f"Sparsity: {self.args.delta_sparsity * 100:.0f}% -> {effective_sparsity * 100:.1f}%")

            # 返回 4 个值 (dw, loss, acc, new_residual)
            return sparse_delta_w, final_loss, final_acc, new_residual

        else:
            # 不稀疏化时，返回空残差
            return delta_w, final_loss, final_acc, {}
