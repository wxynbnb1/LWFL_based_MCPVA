# models/Fed.py

import copy
import torch


def FedAvg_Weighted(w_clients, data_sizes):
    """
    实现 FedAvg 算法的核心加权平均逻辑。
    已根据生产环境要求进行健壮性修正：
    1. 增加模型结构一致性检查。
    2. 排除非浮点张量（如 BatchNorm 的 num_batches_tracked）的平均。
    3. 对 running_mean 和 running_var 进行加权平均。

    参数:
        w_clients (list): 客户端上传的模型权重 (state_dict) 列表。
                          e.g., [w0, w1, w2]
        data_sizes (list): 对应客户端的数据样本数量列表。
                           e.g., [n0, n1, n2]

    返回:
        dict: 经过 FedAvg 聚合后的全局模型权重 (state_dict)。
    """
    if not w_clients:
        return None

    # 0. 健壮性检查：确保所有客户端模型的结构一致
    # 使用第一个客户端的权重作为参考
    w_ref = w_clients[0]
    for i, w_client in enumerate(w_clients):
        # 键一致性检查
        if w_client.keys() != w_ref.keys():
            raise ValueError(f"客户端 {i} 的模型键集合与参考模型不一致。")
        # 形状一致性检查
        for key in w_ref.keys():
            if w_client[key].shape != w_ref[key].shape:
                raise ValueError(f"客户端 {i} 的参数 '{key}' 形状不一致 ({w_client[key].shape} != {w_ref[key].shape})。")

    # 1. 深度复制第一个客户端的权重作为聚合的起点
    w_avg = copy.deepcopy(w_clients[0])

    # 2. 计算总样本数 (用于加权)
    total_data_size = sum(data_sizes)

    # 3. 初始化加权和并累加
    for key in w_avg.keys():

        # --- 关键修正：处理 num_batches_tracked ---
        # BatchNorm 层的 num_batches_tracked 记录了更新次数，不应参与平均。
        if 'num_batches_tracked' in key:
            # 策略：直接采用第一个客户端（服务器全局初始化时）的值，
            # 或者设置为所有客户端的最大值/总和。在 FedAvg 中，通常保持不变。
            # 这里我们选择保持 w_avg 初始副本（即第一个客户端的）的值，不参与后续加权。
            continue

        # --- 关键修正：数据类型安全和加权 ---

        tensor = w_clients[0][key]

        # 检查张量是否为浮点类型 (只对浮点参数进行加权平均)
        if tensor.dtype.is_floating_point:

            # --- 初始化 w_avg[key] 为第一个客户端的加权值 ---
            n0 = data_sizes[0]
            weight_factor_0 = n0 / total_data_size
            w_avg[key].mul_(weight_factor_0)

            # --- 累加剩余客户端的加权值 ---
            for i in range(1, len(w_clients)):
                w_client_i = w_clients[i]
                ni = data_sizes[i]
                weight_factor_i = ni / total_data_size  # ni / N

                # w_avg[key] += w_client_i[key] * (ni / N)
                w_avg[key].add_(w_client_i[key], alpha=weight_factor_i)

        else:
            # 对于非浮点型（如整数，尽管在 RAN 模型中不常见），直接采用第一个客户端的值。
            # 也可以选择采用众数或抛出错误，但保持第一个客户端的值通常是安全的默认行为。
            pass

    return w_avg


# # ---------------------- 修正后的调试示例 ----------------------
# if __name__ == '__main__':
#     print("--- FedAvg_Weighted 修正版独立测试 ---")
#
#     # 模拟客户端权重，包含常规参数和 BN 参数
#     w0 = {
#         'layer1.weight': torch.tensor([1.0, 2.0]),
#         'bn.running_mean': torch.tensor([10.0]),
#         'bn.num_batches_tracked': torch.tensor(10, dtype=torch.long)  # 不应平均
#     }
#     w1 = {
#         'layer1.weight': torch.tensor([3.0, 4.0]),
#         'bn.running_mean': torch.tensor([30.0]),
#         'bn.num_batches_tracked': torch.tensor(20, dtype=torch.long)
#     }
#
#     w_clients_test = [w0, w1]
#     data_sizes_test = [10, 30]
#     total_size = sum(data_sizes_test)  # 40
#
#     # 期望的平均值计算:
#     # layer1.weight[0]: (1.0*10 + 3.0*30) / 40 = (10 + 90) / 40 = 100 / 40 = 2.5
#     # bn.running_mean[0]: (10.0*10 + 30.0*30) / 40 = (100 + 900) / 40 = 1000 / 40 = 25.0
#     # bn.num_batches_tracked: 保持 w0 的值，即 10
#
#     try:
#         w_avg_test = FedAvg_Weighted(w_clients_test, data_sizes_test)
#
#         print(f"客户端数据量: {data_sizes_test}")
#         print(f"总样本数: {total_size}")
#         print("\n--- FedAvg 聚合结果 ---")
#         print(f"layer1.weight: {w_avg_test['layer1.weight']}")
#         print(f"bn.running_mean: {w_avg_test['bn.running_mean']}")
#         print(f"bn.num_batches_tracked: {w_avg_test['bn.num_batches_tracked']}")
#
#         # 验证期望结果
#         assert torch.allclose(w_avg_test['layer1.weight'], torch.tensor([2.5, 3.5]))
#         assert torch.allclose(w_avg_test['bn.running_mean'], torch.tensor([25.0]))
#         # 验证 num_batches_tracked 未参与平均，保留 w0 的值
#         assert w_avg_test['bn.num_batches_tracked'].item() == 10
#         print("\n测试通过！所有修正项验证成功。")
#
#     except ValueError as e:
#         print(f"测试失败：{e}")