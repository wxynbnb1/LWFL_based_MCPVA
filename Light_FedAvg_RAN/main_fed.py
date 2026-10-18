import sys
import os
import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import time  # 【新增】导入时间模块

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

## 获取当前脚本的目录 (通常是项目根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 确保项目根目录在 PATH 中
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 确保 'utils' 目录在 PATH 中 (如果 'utils' 位于根目录下)
utils_path = os.path.join(current_dir, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# ---------------------- 导入自定义模块 (使用绝对导入) ----------------------

# 导入自定义模块
from utils.options import args_parser
from utils.sampling import get_client_data
from models.Nets import RANTabular
from models.Fed import FedAvg_Weighted
from models.Update import LocalUpdate, TabularDataset
from utils.confusion_matrix_plot import plot_and_save_confusion_matrix


# ---------------------- 辅助函数：全局评估 (保持不变) ----------------------
@torch.no_grad()
def test_inference(args, model, test_data, num_classes):
    """
    在全局测试集上评估模型的性能。
    """
    model.eval()
    model.to(args.device)

    loss_func = nn.CrossEntropyLoss()
    total_loss, correct_count, total_count = 0, 0, 0
    y_true, y_pred = [], []

    test_dataset = TabularDataset(test_data['X'], test_data['y'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.local_bs * 2,
        shuffle=False
    )

    for images, labels in test_loader:
        images, labels = images.to(args.device), labels.to(args.device)

        logits = model(images)
        loss = loss_func(logits, labels)

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(1)

        correct_count += (pred == labels).sum().item()
        total_count += labels.size(0)
        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    avg_loss = total_loss / total_count
    avg_acc = correct_count / total_count

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    model.to('cpu')

    return avg_loss, avg_acc, y_true, y_pred


# ---------------------- 辅助函数：随机种子设置 (保持不变) ----------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"--- 随机种子设置为 {seed} ---")


def plot_client_type_distribution(client_data_map, num_users):
    """绘制每个客户端四类人群样本数的横向堆叠柱状图。"""
    type_names = ['Normal', 'Asthmatic', 'Vaper', 'Smoker']
    type_aliases = {name.lower(): name for name in type_names}
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
    counts = np.zeros((num_users, len(type_names)), dtype=int)

    for client_id in range(num_users):
        client = client_data_map.get(client_id)
        if client is None:
            continue
        unknown_types = set()
        for raw_type in client.get('real_type', []):
            canonical = type_aliases.get(str(raw_type).strip().lower())
            if canonical is None:
                unknown_types.add(str(raw_type))
            else:
                counts[client_id, type_names.index(canonical)] += 1
        if unknown_types:
            raise ValueError(
                f"客户端 {client_id} 的 'real type' 列包含未知类别: "
                f"{sorted(unknown_types)}；预期类别为 {type_names}。"
            )

    save_dir = os.path.join(current_dir, 'results_plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'client_type_distribution.png')

    font_size = 14
    fig, ax = plt.subplots(figsize=(12, max(4.8, 1.15 * num_users + 2)), dpi=150)
    y_pos = np.arange(num_users)
    left = np.zeros(num_users, dtype=int)
    for type_idx, (type_name, color) in enumerate(zip(type_names, colors)):
        values = counts[:, type_idx]
        bars = ax.barh(y_pos, values, left=left, height=0.62,
                       label=type_name, color=color, edgecolor='white', linewidth=1.2)
        for client_id, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                ax.text(left[client_id] + value / 2,
                        bar.get_y() + bar.get_height() / 2, str(value),
                        ha='center', va='center', fontsize=font_size,
                        fontweight='bold', color='white')
        left += values

    ax.set_yticks(y_pos, [f'Client {i}' for i in range(num_users)])
    ax.invert_yaxis()
    ax.set_xlabel('No. of Samples', fontsize=font_size)
    ax.set_ylabel('Local Clients', fontsize=font_size)
    # ax.set_title('Distribution of Sample Types Across Clients', fontsize=15, pad=18)
    ax.set_xlim(0, 40)
    ax.grid(axis='x', linestyle='--', alpha=0.28)
    ax.set_axisbelow(True)
    # 显示完整边框：包括 x=40 的右边框和 Client 0 上方的上边框。
    ax.spines[['top', 'right']].set_visible(True)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(ncol=4, loc='lower center',
              bbox_to_anchor=(0.5, 1.01), frameon=False,
              fontsize=font_size)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("\n[Client Distribution] 四类样本分配统计：")
    print(pd.DataFrame(counts, columns=type_names,
                       index=[f'Client {i}' for i in range(num_users)]))
    print(f"[Client Distribution] 柱状图已保存至: {save_path}")
    return save_path


def calculate_pairwise_kl_divergence(client_data_map, num_users, epsilon=1e-10):
    """计算每对客户端四分类分布的双向及对称 KL 散度。"""
    type_names = ['Normal', 'Asthmatic', 'Vaper', 'Smoker']
    type_to_index = {name.lower(): idx for idx, name in enumerate(type_names)}
    counts = np.zeros((num_users, len(type_names)), dtype=np.float64)

    for client_id in range(num_users):
        client = client_data_map.get(client_id)
        if client is None or len(client.get('real_type', [])) == 0:
            raise ValueError(f"客户端 {client_id} 没有可用于计算 KL 散度的样本。")

        unknown_types = set()
        for raw_type in client['real_type']:
            type_idx = type_to_index.get(str(raw_type).strip().lower())
            if type_idx is None:
                unknown_types.add(str(raw_type))
            else:
                counts[client_id, type_idx] += 1
        if unknown_types:
            raise ValueError(
                f"客户端 {client_id} 的 'real type' 列包含未知类别: "
                f"{sorted(unknown_types)}；预期类别为 {type_names}。"
            )

    # 加性平滑避免某客户端缺少某一类别时出现 log(0) 或除以 0。
    probabilities = counts + epsilon
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    results = []
    for client_a in range(num_users):
        for client_b in range(client_a + 1, num_users):
            p = probabilities[client_a]
            q = probabilities[client_b]
            kl_a_to_b = float(np.sum(p * np.log(p / q)))
            kl_b_to_a = float(np.sum(q * np.log(q / p)))
            results.append({
                'client_pair': f'Client {client_a} - Client {client_b}',
                'kl_a_to_b': kl_a_to_b,
                'kl_b_to_a': kl_b_to_a,
                'symmetric_kl': 0.5 * (kl_a_to_b + kl_b_to_a)
            })

    result_df = pd.DataFrame(results)
    save_dir = os.path.join(current_dir, 'results_plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pairwise_client_kl_divergence.csv')
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig', float_format='%.8f')

    print("\n[Pairwise KL Divergence] 客户端两两四分类分布差异（自然对数）：")
    print(result_df.to_string(index=False, float_format=lambda value: f'{value:.8f}'))
    print("说明：symmetric_kl 为两个方向 KL 散度的平均值，数值越大表示分布差异越明显。")
    print(f"[Pairwise KL Divergence] 统计结果已保存至: {save_path}")
    return result_df


# ---------------------- 辅助函数：绘图功能 (已修改) ----------------------
def plot_results(history, args):
    """
    修改说明：
    1. 调整了绘图顺序：先画 Local，再画 Global。
       这样 Global 的 Label 会出现在图例的最下方，且红线会覆盖在最上层。
    """
    actual_rounds = len(history['global_acc'])
    epochs = range(1, actual_rounds + 1)

    save_dir = getattr(args, 'output_dir', './results_plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    alpha_val = getattr(args, 'alpha', 'N/A')
    seed_local_val = getattr(args, 'seed_local', 'N/A')

    # --- 字体大小设置 ---
    FONT_SIZE_TITLE = 14
    FONT_SIZE_LABEL = 20
    FONT_SIZE_TICKS = 18
    FONT_SIZE_LEGEND = 16

    # ========================== 1. 合并绘制 Accuracy 曲线 ==========================
    plt.figure(figsize=(12, 8))

    # 【修改步骤 1】先循环绘制所有 Local Client Accuracy (让它们排在图例上方，且绘图层在下)
    for client_id in range(args.num_users):
        key = f'local_acc_C{client_id}'
        if key in history and history[key]:
            plt.plot(epochs, history[key][:actual_rounds],
                     label=f'Local Client {client_id} Train',
                     linestyle='--', linewidth=2, alpha=0.7)

    # 【修改步骤 2】最后绘制 Global Test Accuracy (让它排在图例最下，且绘图层在最上)
    plt.plot(epochs, history['global_acc'], label='Global Model Test',
             marker='o', linestyle='-', linewidth=2, color='red', alpha=0.9)

    # 1.3 样式调整
    plt.title(f'Accuracy Comparison: Global vs Local (Alpha={alpha_val})', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Communication Rounds', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Accuracy', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=FONT_SIZE_LEGEND, loc='best')

    # 保存
    plot_tag = getattr(args, 'plot_tag', '')
    acc_path = os.path.join(save_dir, f'combined_accuracy{plot_tag}_A{alpha_val}_S{seed_local_val}.png')
    plt.savefig(acc_path)
    plt.close()

    # ========================== 2. 合并绘制 Loss 曲线 ==========================
    plt.figure(figsize=(12, 8))

    # 【修改步骤 3】先循环绘制所有 Local Client Loss
    for client_id in range(args.num_users):
        key = f'local_loss_C{client_id}'
        if key in history and history[key]:
            plt.plot(epochs, history[key][:actual_rounds],
                     label=f'Local Client {client_id} Train',
                     linestyle='--', linewidth=2, alpha=0.7)

    # 【修改步骤 4】最后绘制 Global Test Loss
    plt.plot(epochs, history['global_loss'], label='Global Model Test',
             marker='s', linestyle='-', linewidth=2, color='red', alpha=0.9)

    # 2.3 样式调整
    plt.title(f'Loss Comparison: Global vs Local (Alpha={alpha_val})', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Communication Rounds', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Loss', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=FONT_SIZE_LEGEND, loc='best')

    # 保存
    loss_path = os.path.join(save_dir, f'combined_loss{plot_tag}_A{alpha_val}_S{seed_local_val}.png')
    plt.savefig(loss_path)
    plt.close()

    print(f"\n--- [Updated] 合并结果曲线已保存至: {save_dir} ---")
    print(f"    - Accuracy: {acc_path}")
    print(f"    - Loss:     {loss_path}")


# ---------------------- 主函数：联邦学习循环 (修正与补充) ----------------------

def main_fed():
    # 【新增】开始计时
    start_time = time.time()

    # 1. 初始化参数和环境
    args = args_parser()

    # ================= 【修复报错的核心原因】 =================
    # options.py 中移除了 `--model_name`，但外部脚本（如混淆矩阵工具）依然强依赖此属性。
    # 在不修改 options.py 的前提下，显式为 args 对象绑定缺少的 model_name 属性，确保代码畅通。
    if not hasattr(args, 'model_name'):
        args.model_name = 'RAN'
    # ==========================================================

    setup_seed(args.seed)  # 仅用于全局划分和整体可复现性
    if not hasattr(args, 'alpha'):
        args.alpha = 'N/A'
    # 确保 args.seed_local 存在，即使是从 options.py 来的
    if not hasattr(args, 'seed_local'):
        args.seed_local = args.seed  # 默认使用全局种子

    device = torch.device(args.device)

    # 2. 数据加载与客户端划分
    client_data_map, X_te_global, y_te_global, num_classes, global_scaler = get_client_data(args)
    plot_client_type_distribution(client_data_map, args.num_users)
    calculate_pairwise_kl_divergence(client_data_map, args.num_users)

    # 3. 构造全局测试集字典
    global_test_data = {'X': X_te_global, 'y': y_te_global}

    # 4. 初始化全局模型 (Global Model)
    in_dim = X_te_global.shape[1]

    global_model = RANTabular(
        in_dim=in_dim,
        num_classes=num_classes,
        model_dim=args.model_dim,
        num_blocks=args.num_blocks,
        hidden_mult=args.hidden_mult,
        attn_reduction=args.attn_reduction,
        dropout=args.dropout
    ).to(device)

    global_weights = global_model.state_dict()

    # **【新增】** 通信量统计初始化
    # 计算全模型浮点参数总量 (用于基线通信量)
    total_float_params = sum(
        p.numel() for name, p in global_model.named_parameters()
        if p.dtype.is_floating_point  # 仅统计浮点参数
    )

    # 假设每轮通信中，客户端发送和接收的都是 32 位浮点数 (4 字节)
    BYTES_PER_PARAM = 4

    # 基础通信量 (未稀疏化时，每个客户端每轮发送的参数量)
    base_param_sent_per_client = total_float_params
    # 基础总通信量 (MB) - 单向 (Client -> Server)
    # 理论最大值：所有客户端都参与 * 轮数 * 全模型参数量
    base_comm_mb_per_round = (base_param_sent_per_client * args.num_users * BYTES_PER_PARAM) / (1024 * 1024)

    comm_history = {
        'total_params_sent_per_round': [],  # 每轮聚合时所有客户端发送的总参数量
        'total_comm_mb': 0.0,  # 累计总通信量 (MB)
        'client_participation_rate': []  # 每轮客户端参与率
    }

    print(f"\n[Comm Base] 全模型浮点参数总量: {total_float_params:,}")
    print(f"[Comm Base] **理论**每轮最大通信量 (发送, Full Model, {args.num_users} C): {base_comm_mb_per_round:.2f} MB")

    # 5. 联邦通信循环 (Global Rounds)
    print("\n==================== 启动联邦学习 (FedAvg) 循环 ====================")
    print(f"Non-IID Alpha={args.alpha}, Local Seed={args.seed_local}")
    print(f"参数：Epochs={args.epochs}, Local_BS={args.local_bs}, Users={args.num_users}, Frac={args.frac}")

    # 【修复】定义早停参数，防止 name 'MIN_DELTA' is not defined 错误
    MIN_DELTA = 0.0001
    wait_count = 0
    # PATIENCE = 50 # 如果你需要开启早停逻辑，可以取消注释

    history = {
        'global_loss': [], 'global_acc': [],
        **{f'local_loss_C{i}': [] for i in range(args.num_users)},
        **{f'local_acc_C{i}': [] for i in range(args.num_users)}
    }

    best_global_acc = 0.0
    best_weights = None
    # 核心修正：存储所有达到最高精度的混淆矩阵和对应的轮次
    best_acc_cms = []

    for global_round in range(1, args.epochs + 1):
        local_weights = []
        local_losses = []
        local_data_sizes = []
        # 记录本轮所有客户端的结果，包括未被选中的
        local_round_results = {f'C{i}': {'loss': None, 'acc': None} for i in range(args.num_users)}

        # 客户端选择
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = copy.deepcopy(global_model)
            local_model.load_state_dict(global_weights)

            # 检查客户端是否有数据 (避免 sampling.py 返回空列表)
            if client_data_map[idx]['len'] == 0:
                print(f"  > GR {global_round:02d} | C{idx} 无样本，跳过训练。")
                continue

            local_trainer = LocalUpdate(args=args, client_data=client_data_map[idx])

            # FedAvg 训练返回完整的模型权重 w_local
            w_local, loss_local, acc_local = local_trainer.train(
                model=local_model,
                global_round=global_round
            )

            local_weights.append(w_local)
            local_losses.append(loss_local)
            local_data_sizes.append(client_data_map[idx]['len'])

            # 记录本次训练结果
            local_round_results[f'C{idx}']['loss'] = loss_local
            local_round_results[f'C{idx}']['acc'] = acc_local

            print(
                f"  > GR {global_round:02d} | C{idx} | Samples: {client_data_map[idx]['len']:02d} | Loss: {loss_local:.4f}, Acc: {acc_local * 100:.2f}%")

        # ---------------------- 【新增】本轮通信量计算 ----------------------
        m_actual = len(local_weights)  # 实际参与的客户端数量

        # FedAvg 传输的是完整的模型权重 w_local (或完整的 Delta W)，即 total_float_params
        current_round_sent_params = base_param_sent_per_client * m_actual
        current_round_comm_mb = (current_round_sent_params * BYTES_PER_PARAM) / (1024 * 1024)

        # 累计统计
        comm_history['total_comm_mb'] += current_round_comm_mb
        comm_history['total_params_sent_per_round'].append(current_round_sent_params)
        participation_rate = m_actual / args.num_users if args.num_users > 0 else 0
        comm_history['client_participation_rate'].append(participation_rate)

        # 打印本轮通信统计
        print(
            f"[Comm Stat] GR {global_round:02d} | 参与率: {participation_rate * 100:.0f}% ({m_actual}/{args.num_users})")
        print(f"[Comm Stat] 传输参数量: {current_round_sent_params:,}")
        print(f"[Comm Stat] 本轮通信量: {current_round_comm_mb:.6f} MB")
        print(f"[Comm Stat] 累计总通信量: {comm_history['total_comm_mb']:.2f} MB")

        # 6. 联邦聚合 (仅当有客户端参与训练时才聚合)
        if local_weights:
            global_weights = FedAvg_Weighted(local_weights, local_data_sizes)
            global_model.load_state_dict(global_weights)
        else:
            # 如果本轮没有客户端被选中或有数据，保持模型不变
            print("--- WARNING: 本轮没有客户端参与训练或有数据，全局模型未更新。---")

        # 7. 全局模型评估
        loss_global, acc_global, y_true, y_pred = test_inference(
            args, global_model, global_test_data, num_classes
        )

        history['global_loss'].append(loss_global)
        history['global_acc'].append(acc_global)

        # 7.1. 更新本地历史记录：处理未被选中的客户端
        for client_id in range(args.num_users):
            key_loss = f'local_loss_C{client_id}'
            key_acc = f'local_acc_C{client_id}'

            current_loss = local_round_results[f'C{client_id}']['loss']
            current_acc = local_round_results[f'C{client_id}']['acc']

            if current_loss is not None:
                # 客户端本轮被选中并训练
                history[key_loss].append(current_loss)
                history[key_acc].append(current_acc)
            else:
                # 客户端本轮未被选中 (或无数据被跳过)
                if global_round > 1:
                    # 采用上一轮的本地评估值（平滑曲线）
                    history[key_loss].append(history[key_loss][-1])
                    history[key_acc].append(history[key_acc][-1])
                else:
                    # 第一轮未被选中，记录为 0.0 (或 -1 等占位符)
                    history[key_loss].append(0.0)
                    history[key_acc].append(0.0)

        # 全局结果
        print(
            f"--- Global Round {global_round:02d}/{args.epochs} Completed. Global Loss: {loss_global:.4f} | Global Acc: {acc_global * 100:.2f}% ---")

        # 7.2. 早停机制检查与最佳模型保存
        if acc_global > best_global_acc + MIN_DELTA:
            # 性能有显著提升
            best_global_acc = acc_global
            best_weights = copy.deepcopy(global_weights)
            wait_count = 0

            # 清除旧的混淆矩阵，因为找到了更高的精度
            best_acc_cms = []
            cm = confusion_matrix(y_true, y_pred)
            best_acc_cms.append({'round': global_round, 'cm': cm, 'acc': acc_global})

            print(f"*** NEW BEST ACC: {best_global_acc * 100:.2f}% (Round {global_round}) ***")

        elif acc_global >= best_global_acc:
            # 精度与最高精度相同或仅轻微提升（小于 MIN_DELTA）
            if acc_global == best_global_acc:
                # 精度完全相同，记录该轮次的混淆矩阵
                cm = confusion_matrix(y_true, y_pred)
                best_acc_cms.append({'round': global_round, 'cm': cm, 'acc': acc_global})
                print(f"*** EQUAL BEST ACC: {best_global_acc * 100:.2f}% (Round {global_round}) ***")

            wait_count += 1

        else:
            # 性能下降
            wait_count += 1

    # 8. 训练结束：加载最佳模型、输出最终报告和绘图

    # 8.1. 绘图
    plot_results(history, args)

    # 8.2. 加载最佳模型并评估最终性能
    if best_weights:
        global_model.load_state_dict(best_weights)

    # 重新在全局测试集上运行评估 (确保使用加载的最佳模型)
    final_loss, final_acc, y_true_final, y_pred_final = test_inference(
        args, global_model, global_test_data, num_classes
    )

    # 8.3. 最终报告
    print("\n\n==================== 最终联邦分类报告 (FedAvg) ====================")
    total_rounds = len(history['global_acc'])
    print(f"总通信轮数: {total_rounds} (原计划 {args.epochs})")
    print(f"最佳全局测试精度: {best_global_acc * 100:.2f}%")
    print(f"最终加载模型的测试精度: {final_acc * 100:.2f}% | Loss: {final_loss:.4f}")

    # 定义类别名称
    class_names = getattr(args, 'class_names', ['Normal', 'Abnormal'])

    # 计算最终模型的混淆矩阵
    cm_final = confusion_matrix(y_true_final, y_pred_final)

    # 调用混淆矩阵绘图函数
    plot_and_save_confusion_matrix(
        cm=cm_final,
        class_names=class_names,
        title=' ',
        save_filename=getattr(args, 'cm_filename', 'cm_fedavg_global.png'),
        args=args
    )

    # 输出所有最高准确率对应的混淆矩阵
    if best_acc_cms:
        print("\n\n=============== 所有最高准确率的混淆矩阵 ================")
        print(f"最高精度为: {best_global_acc * 100:.2f}%")

        unique_cms = {}
        for item in best_acc_cms:
            cm_str = np.array_str(item['cm'])
            if cm_str not in unique_cms:
                unique_cms[cm_str] = {'rounds': [item['round']], 'cm': item['cm']}
            else:
                unique_cms[cm_str]['rounds'].append(item['round'])

        for i, cm_info in enumerate(unique_cms.values()):
            rounds_str = ', '.join(map(str, sorted(cm_info['rounds'])))
            print(f"\n--- 混淆矩阵 {i + 1} --- (出现在轮次: {rounds_str})")
            print(cm_info['cm'])

        print("======================================================")
    else:
        print("\n--- 混淆矩阵 (基于最后一轮模型) ---")
        print(cm_final)

    print("\n--- Precision, Recall, F1-Score 报告 (基于最佳模型) ---")
    print(classification_report(y_true_final, y_pred_final, digits=4, target_names=class_names))
    print("======================================================")

    # ---------------------- 【新增】最终通信量统计报告 ----------------------
    final_total_mb = comm_history['total_comm_mb']
    actual_total_params = sum(comm_history['total_params_sent_per_round'])

    # 理论最大总参数量 (全模型 * 总轮数 * 客户端总数)
    base_total_params_max = total_float_params * total_rounds * args.num_users

    comm_reduction_percent = 0.0
    if base_total_params_max > 0:
        # 在 FedAvg 中，通信节省主要来自客户端抽样 (frac < 1)
        comm_reduction_percent = (1 - actual_total_params / base_total_params_max) * 100

    print("\n\n==================== 最终通信量统计报告 (FedAvg) ====================")
    print(f"**通信策略:** 完整模型传输 (FedAvg) | 客户端抽样率: {args.frac * 100:.0f}%")
    print(f"总通信轮数: {total_rounds}")
    print(f"全模型浮点参数总量: {total_float_params:,}")
    print(f"----------------------------------------------------------")
    print(f"实际传输总参数量 (Full Model * 实际参与数): {actual_total_params:,}")
    print(f"累计总通信量 (Client -> Server): **{final_total_mb:.2f} MB**")
    print(f"通信量相比理论最大基线减少: **{comm_reduction_percent:.2f}%** (主要归因于客户端抽样)")
    print("==========================================================")

    # 【新增】停止计时并打印
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n[Timer] FedAvg 训练总耗时: {total_duration:.2f} 秒")

    return {
        'history': history,
        'best_acc': best_global_acc,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'total_rounds': total_rounds,
        'duration_seconds': total_duration,
    }


if __name__ == '__main__':
    try:
        main_fed()
    except Exception as e:
        print(f"\n致命错误：联邦学习过程意外终止。错误信息: {e}")
        sys.exit(1)

# # # ---------------------- 核心函数：联邦学习循环 (重命名并接受 seed) ----------------------
# #
# # 将原 main_fed 逻辑重命名为 run_fedavg_for_seed
# def run_fedavg_for_seed(input_args, local_seed):
#     # 1. 初始化参数和环境
#     # 使用传入的 input_args，并设置 local_seed
#     args = copy.deepcopy(input_args)
#     args.seed_local = local_seed  # 传入循环的 seed
#
#     setup_seed(args.seed)  # 仅用于全局划分和整体可复现性
#     if not hasattr(args, 'alpha'):
#         args.alpha = 'N/A'
#
#     # 【同步修复此处】以防后续你要解开这段注释运行循环模式
#     if not hasattr(args, 'model_name'):
#         args.model_name = 'RAN'
#
#     device = torch.device(args.device)
#
#     # 2. 数据加载与客户端划分
#     # get_client_data 现在使用 args.seed_local
#     client_data_map, X_te_global, y_te_global, num_classes, global_scaler = get_client_data(args)
#
#     # 3. 构造全局测试集字典
#     global_test_data = {'X': X_te_global, 'y': y_te_global}
#
#     # 4. 初始化全局模型 (Global Model)
#     in_dim = X_te_global.shape[1]
#
#     # 注意：每次循环都需要重新初始化模型，确保结果独立
#     global_model = RANTabular(
#         in_dim=in_dim,
#         num_classes=num_classes,
#         model_dim=args.model_dim,
#         num_blocks=args.num_blocks,
#         hidden_mult=args.hidden_mult,
#         attn_reduction=args.attn_reduction,
#         dropout=args.dropout
#     ).to(device)
#
#     global_weights = global_model.state_dict()
#
#     # **【新增】** 早停机制参数 (保持不变)
#     PATIENCE = 50
#     MIN_DELTA = 0.0001
#     wait_count = 0
#
#     # 5. 联邦通信循环 (Global Rounds)
#     print(f"\n======== 启动 FedAvg 循环 (Local Seed: {args.seed_local}) ========")
#     print(f"Non-IID Alpha={args.alpha}, Local Seed={args.seed_local}")
#
#     # ... (history 初始化及 Global Loop 5-7.1 保持不变) ...
#
#     history = {
#         'global_loss': [], 'global_acc': [],
#         **{f'local_loss_C{i}': [] for i in range(args.num_users)},
#         **{f'local_acc_C{i}': [] for i in range(args.num_users)}
#     }
#
#     best_global_acc = 0.0
#     best_weights = None
#     best_acc_cms = []
#
#     for global_round in range(1, args.epochs + 1):
#         # ... (客户端选择、本地训练、聚合等逻辑保持不变) ...
#         local_weights = []
#         local_losses = []
#         local_data_sizes = []
#         local_round_results = {f'C{i}': {'loss': None, 'acc': None} for i in range(args.num_users)}
#
#         m = max(int(args.frac * args.num_users), 1)
#         idxs_users = np.random.choice(range(args.num_users), m, replace=False)
#
#         for idx in idxs_users:
#             local_model = copy.deepcopy(global_model)
#             local_model.load_state_dict(global_weights)
#
#             if client_data_map[idx]['len'] == 0:
#                 continue
#
#             local_trainer = LocalUpdate(args=args, client_data=client_data_map[idx])
#
#             w_local, loss_local, acc_local = local_trainer.train(
#                 model=local_model,
#                 global_round=global_round
#             )
#
#             local_weights.append(w_local)
#             local_losses.append(loss_local)
#             local_data_sizes.append(client_data_map[idx]['len'])
#
#             local_round_results[f'C{idx}']['loss'] = loss_local
#             local_round_results[f'C{idx}']['acc'] = acc_local
#
#         if local_weights:
#             global_weights = FedAvg_Weighted(local_weights, local_data_sizes)
#             global_model.load_state_dict(global_weights)
#
#         loss_global, acc_global, y_true, y_pred = test_inference(
#             args, global_model, global_test_data, num_classes
#         )
#
#         history['global_loss'].append(loss_global)
#         history['global_acc'].append(acc_global)
#
#         # 7.1. 更新本地历史记录 (保持不变)
#         for client_id in range(args.num_users):
#             key_loss = f'local_loss_C{client_id}'
#             key_acc = f'local_acc_C{client_id}'
#
#             current_loss = local_round_results[f'C{client_id}']['loss']
#             current_acc = local_round_results[f'C{client_id}']['acc']
#
#             if current_loss is not None:
#                 history[key_loss].append(current_loss)
#                 history[key_acc].append(current_acc)
#             else:
#                 if global_round > 1:
#                     history[key_loss].append(history[key_loss][-1])
#                     history[key_acc].append(history[key_acc][-1])
#                 else:
#                     history[key_loss].append(0.0)
#                     history[key_acc].append(0.0)
#
#         # 7.2. 早停机制检查与最佳模型保存 (保持不变)
#         if acc_global > best_global_acc + MIN_DELTA:
#             best_global_acc = acc_global
#             best_weights = copy.deepcopy(global_weights)
#             wait_count = 0
#             best_acc_cms = []
#             cm = confusion_matrix(y_true, y_pred)
#             best_acc_cms.append({'round': global_round, 'cm': cm, 'acc': acc_global})
#         elif acc_global >= best_global_acc:
#             if acc_global == best_global_acc:
#                 cm = confusion_matrix(y_true, y_pred)
#                 best_acc_cms.append({'round': global_round, 'cm': cm, 'acc': acc_global})
#             wait_count += 1
#         else:
#             wait_count += 1
#
#         # 7.3. 触发早停
#         if wait_count >= PATIENCE:
#             print(f"!!! 早停触发 !!! Round {global_round} | Best Acc: {best_global_acc * 100:.2f}%")
#             break
#
#     # 8. 返回最终结果
#     # 重新在全局测试集上运行评估 (确保使用加载的最佳模型)
#     if best_weights:
#         global_model.load_state_dict(best_weights)
#
#     final_loss, final_acc, _, _ = test_inference(
#         args, global_model, global_test_data, num_classes
#     )
#
#     # 打印最终报告
#     print(f"--- Local Seed {local_seed} 训练结束。最终 Acc: {final_acc * 100:.2f}%, Loss: {final_loss:.4f} ---")
#
#     # 返回种子、最佳精度和最终损失 (注意：使用最佳精度作为主要指标)
#     return local_seed, best_global_acc, final_loss, len(history['global_acc'])
#
#
# # ---------------------- 新增：循环逻辑 ----------------------
#
# def main():
#     # 1. 初始化参数
#     args = args_parser()
#
#     # 2. 定义循环范围和结果存储
#     START_SEED = 21
#     END_SEED = 100
#
#     # 结果 DataFrame
#     results_df = pd.DataFrame(columns=['local_seed', 'best_test_acc', 'final_test_loss', 'rounds_converged'])
#
#     print(f"\n==================== 启动 Local Seed 循环测试 ({START_SEED} - {END_SEED}) ====================")
#
#     # 3. 循环执行联邦学习
#     for seed in range(START_SEED, END_SEED + 1):
#         try:
#             # 运行 FedAvg，并传入 args 和当前 seed
#             current_seed, acc, loss, rounds = run_fedavg_for_seed(args, seed)
#
#             # 记录结果
#             new_row = pd.DataFrame({
#                 'local_seed': [current_seed],
#                 'best_test_acc': [acc],
#                 'final_test_loss': [loss],
#                 'rounds_converged': [rounds]
#             })
#             results_df = pd.concat([results_df, new_row], ignore_index=True)
#
#             # 实时打印结果
#             print(f"  [SUMMARY] Seed {current_seed}: Best Acc={acc * 100:.2f}%, Final Loss={loss:.4f}")
#
#         except Exception as e:
#             print(f"--- ERROR: Seed {seed} 运行时发生错误: {e} ---")
#
#     # 4. 最终报告和保存
#
#     # 创建保存图表的目录
#     save_dir = './results_plots'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     # 计算统计信息
#     mean_acc = results_df['best_test_acc'].mean()
#     std_acc = results_df['best_test_acc'].std()
#     best_seed_row = results_df.loc[results_df['best_test_acc'].idxmax()]
#
#     print("\n\n==================== Local Seed 循环结果汇总 ====================")
#     print(f"总运行次数: {len(results_df)}")
#     print(f"平均最佳精度: {mean_acc * 100:.2f}% (Std: {std_acc * 100:.2f}%)")
#     print(f"-------------------------------------------------------------")
#     print(f"最佳 Local Seed: {int(best_seed_row['local_seed'])} (Acc: {best_seed_row['best_test_acc'] * 100:.2f}%)")
#     print(f"对应损失: {best_seed_row['final_test_loss']:.4f}")
#     print("===============================================================")
#
#     # 保存为 CSV
#     csv_path = os.path.join(save_dir, 'local_seed_sweep_results_seed71.csv')
#     results_df.to_csv(csv_path, index=False, float_format='%.6f')
#     print(f"\n--- 结果已保存至 CSV 文件: {csv_path} ---")
#
#
# if __name__ == '__main__':
#     try:
#         # 在循环模式下，我们调用新的 main() 函数
#         main()
#     except Exception as e:
#         print(f"\n致命错误：Local Seed 循环过程意外终止。错误信息: {e}")
#         sys.exit(1)
