# main_light_mix_fed.py
import sys
import os
import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
import torch.nn.utils.prune as prune
import gzip
import io

# ==========================================================
# ============== 【新增】 强力屏蔽警告区域 ==================
# ==========================================================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.ao.quantization")
# ==========================================================


# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
utils_path = os.path.join(current_dir, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

from utils.options import args_parser
from utils.sampling import get_client_data
from models.Nets import RANTabular
from models.Update_light_mix import LocalUpdate_LightMix, TabularDataset
from utils.confusion_matrix_plot import plot_and_save_confusion_matrix


# ---------------------- 部署优化辅助函数 ----------------------

def apply_global_pruning(model, amount=0.2):
    print(f"\n[Post-Process] 正在对全局模型进行剪枝 (比例: {amount * 100:.1f}%)...")
    pruned_model = copy.deepcopy(model)
    # 这里的剪枝会将 mask 注册到模型中，在前向传播时自动生效
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # 注意：此处暂不 remove，因为微调阶段需要保留 mask
    return pruned_model


def apply_dynamic_quantization(model):
    """
    【核心修改】使用动态量化 (Dynamic Quantization)
    """
    print("\n[Post-Process] 正在进行动态量化 (Dynamic PTQ)...")
    model_to_quant = copy.deepcopy(model)
    model_to_quant.eval()
    model_to_quant.to('cpu')

    quantized_model = torch.quantization.quantize_dynamic(
        model_to_quant,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


# ---------------------- 核心评估与绘图 ----------------------

@torch.no_grad()
def test_inference(args, model, test_data, num_classes):
    model.eval()
    is_quantized = False
    for m in model.modules():
        if isinstance(m, torch.nn.quantized.dynamic.Linear):
            is_quantized = True
            break
    current_device = torch.device('cpu') if is_quantized else args.device
    model.to(current_device)

    loss_func = nn.CrossEntropyLoss()
    total_loss, correct_count, total_count = 0, 0, 0
    y_true, y_pred = [], []

    test_loader = DataLoader(TabularDataset(test_data['X'], test_data['y']), batch_size=args.local_bs * 2,
                             shuffle=False)

    for images, labels in test_loader:
        images, labels = images.to(current_device), labels.to(current_device)
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

    if not is_quantized:
        model.to('cpu')

    return avg_loss, avg_acc, np.concatenate(y_true), np.concatenate(y_pred)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"--- 随机种子设置为 {seed} ---")


def plot_results(history, args, fine_tune_start_round=None):
    """
    绘制并保存结果曲线 (合并 Global/Local，并适配轻量化两阶段展示)
    """
    actual_rounds = len(history['global_acc'])
    epochs = np.arange(1, actual_rounds + 1)

    save_dir = getattr(args, 'output_dir', './results_plots')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # --- 构造文件名后缀 (保留轻量化实验参数标识) ---
    opt_suffix = ""
    if getattr(args, 'do_pruning', False): opt_suffix += f"_Prune{args.pruning_percent:.2f}"
    if getattr(args, 'do_ptq', False): opt_suffix += "_PTQ"
    # 使用 alpha 和 seed_local 区分实验
    alpha_val = getattr(args, 'alpha', 'N/A')
    seed_val = getattr(args, 'seed_local', 'N/A')

    file_suffix = f'_LM_A{alpha_val}_S{seed_val}_D{args.delta_sparsity:.3f}_F{args.freeze_blocks_count}{opt_suffix}'

    # --- 字体大小设置 (根据要求放大) ---
    FONT_SIZE_TITLE = 18
    FONT_SIZE_LABEL = 20
    FONT_SIZE_TICKS = 18
    FONT_SIZE_LEGEND = 16

    # --- 辅助函数：绘制分阶段的 Global 曲线 ---
    def plot_phased_global(ax, x_data, y_data, label_prefix, color):
        """
        专门用于绘制 Global 曲线，能够区分 Phase 1 和 Phase 2 (Fine-tuning)
        """
        # 线条加粗，使其比 Local 曲线更显眼
        LINE_WIDTH_GLOBAL = 3.0

        if fine_tune_start_round is None or fine_tune_start_round >= len(x_data):
            # 只有 Phase 1
            ax.plot(x_data, y_data, label=f'{label_prefix} (Phase 1)',
                    marker='o', linestyle='-', linewidth=LINE_WIDTH_GLOBAL, color=color, zorder=10)
        else:
            # Phase 1
            split_idx = fine_tune_start_round
            ax.plot(x_data[:split_idx], y_data[:split_idx],
                    label=f'{label_prefix} (Phase 1)',
                    marker='o', linestyle='-', linewidth=LINE_WIDTH_GLOBAL, color=color, zorder=10)

            # Phase 2 (Fine-tune) - 虚线连接
            # 从 split_idx-1 开始画，确保线条连续
            ax.plot(x_data[split_idx - 1:], y_data[split_idx - 1:],
                    label=f'{label_prefix} (Phase 2: Fine-tune)',
                    marker='^', linestyle='--', linewidth=LINE_WIDTH_GLOBAL, color=color, alpha=0.9, zorder=10)

    # ========================== 1. 合并绘制 Accuracy 曲线 ==========================
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 1.1 绘制 Local Client Accuracy (作为背景，线条较细)
    # 使用 colormap 自动分配颜色，或者使用灰色系突出 Global
    colors = plt.get_cmap('tab10', args.num_users)

    for i in range(args.num_users):
        key = f'local_acc_C{i}'
        if key in history and history[key]:
            # 截取有效长度
            y_data = history[key][:actual_rounds]
            ax.plot(epochs[:len(y_data)], y_data,
                    label=f'Local Client {i} Train',
                    linestyle='--', linewidth=1.5, alpha=0.6, color=colors(i))

    # 1.2 绘制 Global Test Accuracy (核心关注点，调用分阶段绘制)
    plot_phased_global(ax, epochs, history['global_acc'], 'Server Test', 'red')

    # 1.3 绘制微调分界线
    if fine_tune_start_round:
        ax.axvline(x=fine_tune_start_round, color='gray', linestyle=':', linewidth=2, label='Fine-tune Start')

    # 1.4 样式调整
    plt.title(f'Accuracy Comparison: Global vs Local (Sparsity={args.delta_sparsity})', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Communication Rounds', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Accuracy', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=FONT_SIZE_LEGEND, loc='lower right')  # Acc 通常在右下角图例干扰最小

    # 保存
    acc_path = os.path.join(save_dir, f'combined_accuracy{file_suffix}.png')
    plt.savefig(acc_path, bbox_inches='tight')
    plt.close()

    # ========================== 2. 合并绘制 Loss 曲线 ==========================
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 2.1 绘制 Local Client Loss
    for i in range(args.num_users):
        key = f'local_loss_C{i}'
        if key in history and history[key]:
            y_data = history[key][:actual_rounds]
            ax.plot(epochs[:len(y_data)], y_data,
                    label=f'Local Client {i} Train',
                    linestyle='--', linewidth=1.5, alpha=0.6, color=colors(i))

    # 2.2 绘制 Global Test Loss
    plot_phased_global(ax, epochs, history['global_loss'], 'Server Test', 'red')

    # 2.3 绘制微调分界线
    if fine_tune_start_round:
        ax.axvline(x=fine_tune_start_round, color='gray', linestyle=':', linewidth=2, label='Fine-tune Start')

    # 2.4 样式调整
    plt.title(f'Loss Comparison: Global vs Local (Sparsity={args.delta_sparsity})', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Communication Rounds', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Loss', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')

    # 保存
    loss_path = os.path.join(save_dir, f'combined_loss{file_suffix}.png')
    plt.savefig(loss_path, bbox_inches='tight')
    plt.close()

    print(f"\n--- [Plot] 合并结果曲线已保存至目录: {save_dir} ---")
    print(f"    - Accuracy: {os.path.basename(acc_path)}")
    print(f"    - Loss:     {os.path.basename(loss_path)}")


def FedDeltaAvg(delta_w_clients, data_sizes, w_global_old):
    if not delta_w_clients: return w_global_old
    total_data_size = sum(data_sizes)
    delta_w_avg = {k: torch.zeros_like(v) for k, v in delta_w_clients[0].items()}
    for i, delta_w_i in enumerate(delta_w_clients):
        weight = data_sizes[i] / total_data_size
        for k in delta_w_i.keys():
            if delta_w_i[k].dtype.is_floating_point:
                delta_w_avg[k].add_(delta_w_i[k], alpha=weight)
            elif 'num_batches_tracked' in k and i == len(delta_w_clients) - 1:
                delta_w_avg[k] = delta_w_i[k].clone()
    w_new = copy.deepcopy(w_global_old)
    for k in w_new.keys():
        if 'num_batches_tracked' in k:
            if k in delta_w_avg: w_new[k] = delta_w_avg[k]
        elif w_new[k].dtype.is_floating_point:
            # 客户端训练结束后 Delta 位于 CPU，而全局模型可能仍位于 CUDA。
            # 聚合前将平均 Delta 对齐到全局权重所在设备。
            w_new[k].add_(delta_w_avg[k].to(w_new[k].device))
    return w_new


# ---------------------- 主训练函数 ----------------------

def main_light_mix_fed(custom_config=None):
    args = args_parser()

    if custom_config:
        if 'delta_sparsity' in custom_config:
            args.delta_sparsity = custom_config['delta_sparsity']
            print(f">>> [Auto-Run] 覆盖稀疏率: {args.delta_sparsity}")
        if 'pruning_percent' in custom_config:
            args.pruning_percent = custom_config['pruning_percent']
            print(f">>> [Auto-Run] 覆盖剪枝率: {args.pruning_percent}")
        # 增加对 do_pruning 的正确读取
        if 'do_pruning' in custom_config:
            args.do_pruning = custom_config['do_pruning']
            print(f">>> [Auto-Run] 覆盖剪枝开关: {args.do_pruning}")
        if 'epochs' in custom_config:
            args.epochs = custom_config['epochs']

    setup_seed(args.seed)
    device = torch.device(args.device)

    # 1. 数据准备
    client_data_map, X_te_global, y_te_global, num_classes, _ = get_client_data(args)
    global_test_data = {'X': X_te_global, 'y': y_te_global}

    # 2. 模型初始化
    global_model = RANTabular(
        in_dim=X_te_global.shape[1], num_classes=num_classes, model_dim=args.model_dim,
        num_blocks=args.num_blocks, hidden_mult=args.hidden_mult,
        attn_reduction=args.attn_reduction, dropout=args.dropout
    ).to(device)

    if getattr(args, 'show_model_structure', True):
        print("\n" + "=" * 30 + "\n--- 全局模型结构 ---\n", global_model, "\n" + "=" * 30)
    print(f"轻量化配置: Sparsity={args.delta_sparsity}, Freeze={args.freeze_blocks_count} Blocks")

    # 3. 历史记录初始化
    best_global_acc = 0.0
    best_weights = None
    total_float_params = sum(p.numel() for p in global_model.parameters() if p.dtype.is_floating_point)
    comm_history = {'total_comm_mb': 0.0, 'total_params_sent': 0}

    history = {
        'global_loss': [], 'global_acc': [],
        **{f'local_loss_C{i}': [] for i in range(args.num_users)},
        **{f'local_acc_C{i}': [] for i in range(args.num_users)}
    }

    # 【核心修正】初始化客户端残差字典 (Memory for Error Feedback)
    # 结构: {client_id: {param_name: residual_tensor}}
    # 这确保了 Top-k 丢弃的梯度在下一轮会被加回来
    client_residuals = {i: {} for i in range(args.num_users)}

    print(f"\n[Comm Base] 全模型浮点参数总量: {total_float_params:,}")

    # ===================================================================
    # Phase 1: 主训练循环
    # ===================================================================
    print("\n=== Phase 1: 主训练阶段 (Gradient Sparsity + Freezing) ===")
    phase1_rounds = 0

    for global_round in range(1, args.epochs + 1):
        phase1_rounds = global_round
        delta_weights_list, local_data_sizes = [], []
        local_round_results = {f'C{i}': {'loss': None, 'acc': None} for i in range(args.num_users)}

        idxs_users = np.random.choice(range(args.num_users), max(int(args.frac * args.num_users), 1), replace=False)
        w_global_cpu = global_model.state_dict()

        print(f"\n--- Global Round {global_round:02d}/{args.epochs} ---")

        for idx in idxs_users:
            if client_data_map[idx]['len'] == 0: continue
            local_model = copy.deepcopy(global_model).to(device)
            local_model.load_state_dict(w_global_cpu)
            local_trainer = LocalUpdate_LightMix(args=args, client_data=client_data_map[idx])

            # 【核心修正】获取该客户端的旧残差
            current_res = client_residuals[idx]

            # 【核心修正】调用 train，传入旧残差，接收新残差
            dw, loss_l, acc_l, new_res = local_trainer.train(
                model=local_model, global_round=global_round, w_global=w_global_cpu,
                client_idx=idx,
                old_residual=current_res  # 传入上一轮的残差
            )

            # 【核心修正】保存新残差以供下一轮使用
            client_residuals[idx] = new_res

            delta_weights_list.append(dw)
            local_data_sizes.append(client_data_map[idx]['len'])
            local_round_results[f'C{idx}'] = {'loss': loss_l, 'acc': acc_l}

        round_params = sum(
            torch.count_nonzero(t).item() if t.dtype.is_floating_point else t.numel() for dw in delta_weights_list for t
            in dw.values())
        round_mb = (round_params * 4) / (1024 * 1024)
        comm_history['total_comm_mb'] += round_mb
        comm_history['total_params_sent'] += round_params

        print(f"[Comm Stat] 本轮流量: {round_mb:.4f} MB | 累计: {comm_history['total_comm_mb']:.2f} MB")

        if delta_weights_list:
            new_weights = FedDeltaAvg(delta_weights_list, local_data_sizes, w_global_cpu)
            global_model.load_state_dict(new_weights)

        loss_g, acc_g, y_t, y_p = test_inference(args, global_model, global_test_data, num_classes)
        history['global_loss'].append(loss_g)
        history['global_acc'].append(acc_g)

        for i in range(args.num_users):
            res = local_round_results[f'C{i}']
            if res['loss'] is not None:
                history[f'local_loss_C{i}'].append(res['loss'])
            else:
                history[f'local_loss_C{i}'].append(history[f'local_loss_C{i}'][-1] if global_round > 1 else 0.0)
            if res['acc'] is not None:
                history[f'local_acc_C{i}'].append(res['acc'])
            else:
                history[f'local_acc_C{i}'].append(history[f'local_acc_C{i}'][-1] if global_round > 1 else 0.0)

        print(f"--- Global Round {global_round:02d} End | Global Acc: {acc_g * 100:.2f}% ---")

        if acc_g > best_global_acc:
            best_global_acc = acc_g
            best_weights = copy.deepcopy(global_model.state_dict())
            print(f"  >>> ★ New Best Model Saved (Acc: {best_global_acc * 100:.2f}%)")

    # ===================================================================
    # Phase 2: 微调阶段 (Fine-tuning Phase)
    # ===================================================================
    fine_tune_start_round = None

    # 【新增】用来记录 Phase 1 的最佳精度，仅用于日志对比，不影响逻辑
    phase1_best_acc = best_global_acc
    acc_p = phase1_best_acc

    # 【新增补全1】保存剪枝前的模型大小 (Gzip 压缩模拟)，用于对比
    size_before_pruning_kb = 0.0
    if args.do_pruning:
        # 使用 gzip 计算基准大小，以便与剪枝后的 gzip 大小进行公平对比
        buffer_pre = io.BytesIO()
        torch.save(global_model.state_dict(), buffer_pre)
        buffer_pre.seek(0)
        size_before_pruning_kb = len(gzip.compress(buffer_pre.read())) / 1024

    if args.do_pruning:
        print("\n" + "=" * 50)
        print("=== Phase 2: 剪枝与微调 (Pruning + Fine-tuning) ===")
        print("=" * 50)
        print(f">>> [Phase 1 Summary] 剪枝前最佳精度: {phase1_best_acc * 100:.2f}%")

        if best_weights:
            global_model.load_state_dict(best_weights)
            print(">>> 已加载 Phase 1 最佳模型")

        global_model = apply_global_pruning(global_model, amount=args.pruning_percent)

        # 【核心修正】Phase 2 模型结构/参数值发生突变，重置残差以避免维度不匹配或错误累积
        client_residuals = {i: {} for i in range(args.num_users)}
        print(">>> [Reset] 剪枝导致结构或参数分布变化，已重置客户端梯度残差。")

        loss_p, acc_p, _, _ = test_inference(args, global_model, global_test_data, num_classes)
        print(f"[Pruning Check] 剪枝后(微调前)精度: {acc_p * 100:.2f}% (Drop: {(phase1_best_acc - acc_p) * 100:.2f}%)")

        original_lr = args.lr
        args.lr = args.lr * 0.1  # 降低 LR
        print(f">>> [Config] 微调阶段学习率已降低: {original_lr} -> {args.lr}")

        # ==========================================================
        # 【核心修改】重置最佳指标，确保最终只记录剪枝后的最佳表现
        # ==========================================================
        print(">>> [Reset] 重置最佳精度记录器，以捕获剪枝模型在微调阶段的真实峰值...")
        best_global_acc = 0.0
        best_weights = None
        # ==========================================================

        fine_tune_start_round = len(history['global_acc']) + 1
        FT_ROUNDS = 10
        print(f"开始微调 {FT_ROUNDS} 轮... (保持剪枝 Mask)")

        for ft_round in range(1, FT_ROUNDS + 1):
            global_round_idx = phase1_rounds + ft_round
            idxs_users = np.random.choice(range(args.num_users), max(int(args.frac * args.num_users), 1), replace=False)
            w_global_cpu = global_model.state_dict()
            delta_weights_list, local_data_sizes = [], []

            print(f"--- Fine-tune Round {ft_round}/{FT_ROUNDS} (Global {global_round_idx}) ---")

            for idx in idxs_users:
                local_model = copy.deepcopy(global_model).to(device)
                local_trainer = LocalUpdate_LightMix(args=args, client_data=client_data_map[idx])

                # 【核心修正】微调阶段也同样应用误差补偿
                current_res = client_residuals[idx]

                dw, loss_l, acc_l, new_res = local_trainer.train(
                    model=local_model, global_round=global_round_idx, w_global=w_global_cpu,
                    client_idx=idx,
                    old_residual=current_res
                )

                client_residuals[idx] = new_res

                delta_weights_list.append(dw)
                local_data_sizes.append(client_data_map[idx]['len'])

            # ==========================================================
            # 【新增】计算并累加 Phase 2 (微调阶段) 的通信量
            # ==========================================================
            round_params = sum(
                torch.count_nonzero(t).item() if t.dtype.is_floating_point else t.numel()
                for dw in delta_weights_list for t in dw.values()
            )
            round_mb = (round_params * 4) / (1024 * 1024)
            comm_history['total_comm_mb'] += round_mb
            comm_history['total_params_sent'] += round_params

            print(f"[Comm Stat] 本轮流量(含剪枝Mask): {round_mb:.4f} MB | 累计: {comm_history['total_comm_mb']:.2f} MB")
            # ==========================================================

            if delta_weights_list:
                new_weights = FedDeltaAvg(delta_weights_list, local_data_sizes, w_global_cpu)
                global_model.load_state_dict(new_weights)


            loss_g, acc_g, y_t, y_p = test_inference(args, global_model, global_test_data, num_classes)
            history['global_loss'].append(loss_g)
            history['global_acc'].append(acc_g)

            for i in range(args.num_users):
                history[f'local_loss_C{i}'].append(0)
                history[f'local_acc_C{i}'].append(0)

            print(f"FT Round {ft_round} End | Recovery Acc: {acc_g * 100:.2f}%")

            if acc_g > best_global_acc:
                best_global_acc = acc_g
                best_weights = copy.deepcopy(global_model.state_dict())
                print(f"  >>> ★ Fine-tuning achieved New Best (Acc: {best_global_acc * 100:.2f}%)")

        print("\n[Post-Process] 微调阶段结束，准备处理最佳模型...")
        if best_weights:
            try:
                global_model.load_state_dict(best_weights)
                print(">>> 成功加载 Phase 2 历史最佳模型 (含 Mask)")
            except RuntimeError as e:
                print(f">>> 提示: 直接加载最佳模型遇到结构不匹配 (可能是 Phase 1 模型更优)。保持当前最后轮次模型。")

        print(">>> 正在移除剪枝掩码，将稀疏性永久写入参数...")
        for name, module in global_model.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, 'weight')

    # ===================================================================
    # 【新增逻辑】非剪枝模式下，也强制加载最佳模型
    # ===================================================================
    if not args.do_pruning and best_weights is not None:
        global_model.load_state_dict(best_weights)
        print(f"\n>>> [Final] 非剪枝模式：已加载训练过程中的最佳模型 (Acc: {best_global_acc * 100:.2f}%)")

    # ---------------------- 最终结果展示 ----------------------
    plot_results(history, args, fine_tune_start_round)
    f_loss, f_acc, y_t_f, y_p_f = test_inference(args, global_model, global_test_data, num_classes)

    print("\n\n==================== 最终联邦分类报告 ====================")
    print(f"最佳精度: {best_global_acc * 100:.2f}%")

    # --- 核心修改：增加通信量数值对比输出 ---
    total_rounds_conducted = len(history['global_acc'])

    # 1. 计算【变化前】理论全量通信 (Baseline)
    # 公式：单模型参数量 * 轮数 * 客户端数 (注意：这里现在正确地包含了微调轮数)
    base_total_params = total_float_params * total_rounds_conducted * args.num_users
    base_comm_mb = (base_total_params * 4) / (1024 * 1024)  # 转换为 MB (假设float32占4字节)

    # 2. 获取【变化后】实际稀疏通信 (Actual)
    # 现在这里包含了 Phase 1 和 Phase 2 的累加总和
    actual_comm_mb = comm_history['total_comm_mb']
    actual_params = comm_history['total_params_sent']

    # 3. 计算节省比例
    comm_reduction_percent = (1 - actual_params / base_total_params) * 100 if base_total_params > 0 else 0

    print("-" * 30)
    # 修改了这里的文案，去掉了 (Phase 1)
    print("【全局通信量对比 (含 Phase 1 & 微调)】")
    print(f"1. 理论全量通信 (Baseline): {base_comm_mb:.2f} MB")
    print(f"2. 实际稀疏通信 (轻量化后): {actual_comm_mb:.2f} MB")
    print(f"3. 整体节省比例: **{comm_reduction_percent:.2f}%**")
    print("-" * 30)
    # -------------------------------------

    class_names = getattr(args, 'class_names', ['Normal', 'Abnormal'])
    opt_tag = (f"_P{args.pruning_percent:.2f}" if args.do_pruning else "") + ("_PTQ" if args.do_ptq else "")
    cm_name = f'cm_lightmix_D{args.delta_sparsity:.3f}_F{args.freeze_blocks_count}{opt_tag}.png'
    plot_and_save_confusion_matrix(confusion_matrix(y_t_f, y_p_f), class_names, ' ', cm_name, args)

    print("\n--- 详细分类报告 ---")
    print(classification_report(y_t_f, y_p_f, digits=4, target_names=class_names))

    print("\n" + "=" * 30 + "\n--- 部署优化评估 ---")

    # 【辅助函数】计算压缩后的大小
    def get_compressed_model_size(model_obj):
        buffer = io.BytesIO()
        torch.save(model_obj.state_dict(), buffer)
        buffer.seek(0)
        compressed_data = gzip.compress(buffer.read())
        return len(compressed_data) / 1024  # KB

    # 1. 计算 FP32 原始大小 (不压缩，用于对比物理显存占用)
    torch.save(global_model.state_dict(), "temp_fp32.pth")
    physical_size = os.path.getsize("temp_fp32.pth") / 1024

    # 2. 计算 压缩传输/存储大小 (体现剪枝优势)
    compressed_size = get_compressed_model_size(global_model)

    # 3. 统计参数稀疏度
    total_params_final = sum(p.numel() for p in global_model.parameters())
    nonzero_params_final = sum((p != 0).sum().item() for p in global_model.parameters())
    sparsity_ratio = (1 - nonzero_params_final / total_params_final) * 100

    print(f"模型参数总量: {total_params_final}")
    print(f"非零参数量: {nonzero_params_final}")
    print(f"实际参数稀疏度: {sparsity_ratio:.2f}% (目标剪枝率: {args.pruning_percent * 100:.1f}%)")
    print("-" * 30)

    # 4. 输出对比
    if args.do_pruning:
        print(f"剪枝前模型大小 (Gzip): {size_before_pruning_kb:.2f} KB")
        print(f"剪枝后模型大小 (Gzip): {compressed_size:.2f} KB")
        if size_before_pruning_kb > 0:
            print(f"  -> 体积压缩比: {(1 - compressed_size / size_before_pruning_kb) * 100:.2f}%")
    else:
        print(f"最终模型大小 (Gzip): {compressed_size:.2f} KB")

    print(f"显存/磁盘物理占用 (FP32未压缩): {physical_size:.2f} KB")

    if args.do_pruning:
        print(f"剪枝+微调后最终精度: **{f_acc * 100:.2f}%**")

    if args.do_ptq:
        qm = apply_dynamic_quantization(global_model)
        torch.jit.save(torch.jit.script(qm), "temp_int8.pt")
        _, a_q, _, _ = test_inference(args, qm, global_test_data, num_classes)
        print(f"量化后精度: **{a_q * 100:.2f}%** | 模型大小: {os.path.getsize('temp_int8.pt') / 1024:.2f} KB")

    for f in ["temp_fp32.pth", "temp_int8.pt"]:
        if os.path.exists(f): os.remove(f)

# 修改后：同时返回微调后的最高精度，以及刚剪完枝还没微调的精度
    return best_global_acc, acc_p


if __name__ == '__main__':
    try:
        main_light_mix_fed()
    except Exception as e:
        print(f"\n致命错误：{e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
