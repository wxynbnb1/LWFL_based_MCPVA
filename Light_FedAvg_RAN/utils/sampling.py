# utils/sampling.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


# ---------------------- 1. 数据加载与预处理 (保持不变) ----------------------

def get_full_data(args, return_real_types=False):
    """
    加载完整的全局数据，进行标签编码和标准化。
    **使用 args.seed 确保全局训练集/测试集划分固定。**
    **标签手动映射：'normal' -> 0, 'EO' -> 1。**
    """
    excel_path = args.excel_path
    label_col = args.label_column
    feature_cols = args.feature_cols
    seed = args.seed  # 全局种子，固定划分

    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
    except FileNotFoundError:
        print(f"错误：文件未找到在 {excel_path}。")
        raise

    real_type_col = "real type"
    required_cols = feature_cols + [label_col]
    if return_real_types:
        required_cols.append(real_type_col)
    df = df.dropna(subset=required_cols)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y_raw = df[label_col].to_numpy()

    # >>>>>>>>>>>>>>>>> 标签编码：手动映射修改开始 <<<<<<<<<<<<<<<<<
    # 定义手动映射字典
    label_map = {'Normal': 0, 'Abnormal': 1}

    # 检查原始数据中的标签是否都在预期映射中
    unique_labels = np.unique(y_raw)
    for label in unique_labels:
        if label not in label_map:
            raise ValueError(
                f"错误：原始标签 '{label}' 不在预期的手动映射中 ({list(label_map.keys())})。"
            )

    # 应用手动映射
    y = np.array([label_map[label] for label in y_raw])

    # 手动定义 classes 和 num_classes 以保持后续代码兼容性
    # 确保 'normal' -> 0, 'EO' -> 1 的顺序
    classes = np.array(['Normal', 'Abnormal'])
    num_classes = len(classes)
    # >>>>>>>>>>>>>>>>> 标签编码：手动映射修改结束 <<<<<<<<<<<<<<<<<

    # 划分出一个全局测试集（使用 args.seed）
    # 注意：stratify=y 依赖于 y 是数值标签 (0, 1)
    if return_real_types:
        real_types = df[real_type_col].astype(str).str.strip().to_numpy()
        (X_tr_fed, X_te_global, y_tr_fed, y_te_global,
         real_types_tr_fed, _) = train_test_split(
            X, y, real_types, test_size=args.test_size,
            stratify=y, random_state=seed
        )
    else:
        X_tr_fed, X_te_global, y_tr_fed, y_te_global = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=seed
        )

    # 对联邦训练集进行标准化拟合
    scaler = StandardScaler().fit(X_tr_fed)

    # 对联邦训练集和全局测试集进行转换
    X_tr_fed_scaled = scaler.transform(X_tr_fed)
    X_te_global_scaled = scaler.transform(X_te_global)

    print(f"\n[Data Prep] 原始数据总量: {len(X)} 样本")
    print(f"[Data Prep] 联邦训练集总量: {len(X_tr_fed_scaled)} 样本")
    print(f"[Data Prep] 全局测试集总量: {len(X_te_global_scaled)} 样本")
    print(f"[Data Prep] 类别数: {num_classes}, 特征维度: {len(feature_cols)}")

    # 返回值不变，保持兼容
    result = (X_tr_fed_scaled, y_tr_fed, X_te_global_scaled,
              y_te_global, num_classes, scaler)
    if return_real_types:
        return result + (real_types_tr_fed,)
    return result


# ---------------------- 2. IID (独立同分布) 划分 ----------------------

def iid_split(y_data, num_users, seed):
    """
    IID 划分: 随机打乱索引，并平均分配给客户端。
    """
    np.random.seed(seed)
    num_samples = len(y_data)
    # 随机打乱所有样本的索引
    all_indices = np.random.permutation(num_samples)

    # 均匀分配
    client_indices = defaultdict(lambda: [])
    split_size = num_samples // num_users

    for i in range(num_users):
        start = i * split_size
        end = start + split_size
        client_indices[i].extend(all_indices[start:end])

    # 将剩余的样本随机分配给客户端
    remainder = num_samples % num_users
    if remainder > 0:
        remaining_indices = all_indices[num_users * split_size:]
        np.random.shuffle(remaining_indices)
        for i, idx in enumerate(remaining_indices):
            client_indices[i % num_users].append(idx)

    return client_indices


# ---------------------- 3. Non-IID Dirichlet 划分 (引入 seed_local) ----------------------

def non_iid_split_dirichlet(y_data, num_users, alpha, seed_local):
    """
    使用 Dirichlet 分布 (Dirichlet(alpha)) 来划分 Non-IID 数据。

    **使用 args.seed_local 保证 Non-IID 划分的随机性与全局测试集划分独立。**

    参数:
        y_data (np.array): 标签数组。
        num_users (int): 客户端数量 (即 args.num_users)。
        alpha (float): Dirichlet 分布参数。
        seed_local (int): 本地划分的随机种子。

    返回:
        dict: 键为客户端ID，值为该客户端持有的数据索引列表。
    """
    # 使用 seed_local 控制本地划分
    np.random.seed(seed_local)

    # 获取类别和类别数量
    classes = np.unique(y_data)
    num_classes = len(classes)

    # 将标签映射到对应的索引列表
    label_to_indices = defaultdict(lambda: [])
    for idx, label in enumerate(y_data):
        label_to_indices[label].append(idx)

    client_indices = defaultdict(lambda: [])
    total_samples = len(y_data)

    # --- 核心划分逻辑 ---
    for class_id in range(num_classes):
        class_indices = label_to_indices[class_id]
        class_size = len(class_indices)

        # 1. 从 Dirichlet 分布中采样比例
        # 确保比例总和为 1.0
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))

        # 2. 根据比例计算每个客户端应该获取的样本数量
        # np.round: 四舍五入，并强制将总和调整为 class_size
        sample_counts = np.round(proportions * class_size).astype(int)

        # 调整总数，确保总样本数等于 class_size
        diff = class_size - np.sum(sample_counts)
        # 将差值随机（或按比例）分配给客户端
        if diff != 0:
            # 找到比例最大的客户端索引，进行调整
            user_to_adjust = np.random.choice(num_users, size=abs(diff),
                                              p=proportions / proportions.sum() if proportions.sum() else None)
            for user_id in user_to_adjust:
                sample_counts[user_id] += (1 if diff > 0 else -1)

        # 确保分配后总数精确
        if np.sum(sample_counts) != class_size:
            # 兜底：如果还不对，直接在第一个客户端上调整
            sample_counts[0] += (class_size - np.sum(sample_counts))

        # 3. 随机打乱当前类别的索引并分配给客户端
        np.random.shuffle(class_indices)  # 确保每次划分时，同一类别的样本是随机分配的

        current_idx = 0
        for user_id in range(num_users):
            count = sample_counts[user_id]
            # 分配索引
            client_indices[user_id].extend(class_indices[current_idx: current_idx + count])
            current_idx += count

    # 4. 验证和报告
    total_split_indices = sum(len(v) for v in client_indices.values())
    if total_split_indices != total_samples:
        print(f"[ERROR] 划分后的总样本数 ({total_split_indices}) 不等于原始总样本数 ({total_samples})")

    print(f"\n[Sampling] Non-IID 数据划分完成 (Dirichlet Alpha={alpha}, Seed={seed_local})")
    print(f"[Sampling] 客户端总数: {num_users}")
    for user_id in range(num_users):
        user_y = y_data[client_indices[user_id]]
        class_counts = np.bincount(user_y, minlength=num_classes)
        print(f"  - 客户端 {user_id}: 样本数={len(user_y)}, 标签分布={class_counts}")

    return client_indices


# ---------------------- 4. 主函数 (供 main_fed.py 调用) ----------------------

def get_client_data(args):
    """
    联邦学习数据准备的入口函数。
    根据 args.iid 选择 IID 或 Non-IID 划分。
    """
    # 1. 加载和预处理全局数据
    (X_tr_fed, y_tr_fed, X_te_global, y_te_global, num_classes,
     scaler, real_types_tr_fed) = get_full_data(args, return_real_types=True)

    # 2. 划分客户端训练集索引
    if args.iid:
        print(f"\n[Sampling] 采用 IID (独立同分布) 划分。")
        client_indices = iid_split(
            y_tr_fed,
            args.num_users,
            seed=args.seed_local  # IID 划分也使用 seed_local
        )
    else:
        # 默认 Non-IID (医院数据不互通，采用 Non-IID 更符合实际)
        client_indices = non_iid_split_dirichlet(
            y_tr_fed,
            args.num_users,
            alpha=args.alpha,
            seed_local=args.seed_local  # Non-IID 划分使用 seed_local
        )

    # 3. 构造客户端数据集对象
    client_datasets = {}
    for i in range(args.num_users):
        indices = client_indices[i]

        # 确保客户端至少有样本
        if len(indices) == 0:
            print(f"[WARNING] 客户端 {i} 在划分后没有分配到训练样本。")
            continue

        # 根据索引切分出该客户端的 X 和 y
        client_X = X_tr_fed[indices]
        client_y = y_tr_fed[indices]

        # 返回 NumPy 数据子集
        client_datasets[i] = {
            'X': client_X,
            'y': client_y,
            'real_type': real_types_tr_fed[indices],
            'len': len(client_X)
        }

    # 4. 返回所有必要组件
    return client_datasets, X_te_global, y_te_global, num_classes, scaler


# ---------------------- 5. 调试示例 (可选) ----------------------
if __name__ == '__main__':
    # 这是一个独立的测试块，需要模拟 args 对象
    class MockArgs:
        def __init__(self, iid=False, alpha=0.5):
            # 这里的 excel_path 需要手动替换成你的真实路径，仅用于测试
            self.excel_path = r"D:\AWXY\研究生\小论文\数据1+3.xlsx"  # 请确保替换为你的实际路径
            self.label_column = "类型"
            self.feature_cols = ["TV", "COV", "GI", "RVDI"]
            self.test_size = 0.20
            self.num_users = 3
            self.seed = 167  # 全局测试集固定种子
            self.seed_local = 222  # 本地划分种子
            self.iid = iid
            self.alpha = alpha


    # --- 测试 Non-IID 划分 ---
    args_non_iid = MockArgs(iid=False, alpha=0.5)
    print("--- 运行 sampling.py Non-IID (alpha=0.5) 测试 ---")
    client_data_non_iid, X_te, y_te, num_classes, scaler = get_client_data(args_non_iid)

    print("\n[DEBUG] 全局测试集大小:", len(X_te))

    # --- 测试 IID 划分 ---
    args_iid = MockArgs(iid=True)
    print("\n--- 运行 sampling.py IID 测试 ---")
    client_data_iid, _, _, _, _ = get_client_data(args_iid)
