import argparse
import torch


def args_parser():
    """
    解析命令行参数。
    包含了模型结构、训练超参数和联邦学习参数。
    """
    parser = argparse.ArgumentParser(description='Federated Learning for RAN, FT-T, and TabM Tabular Network')

    # ===================================================================
    # 1. 联邦学习 (FedAvg) 参数
    # ===================================================================
    parser.add_argument('--epochs', type=int, default=100, help="联邦通信轮数 (Global rounds of training)")
    parser.add_argument('--num_users', type=int, default=3, help="参与联邦学习的客户端/医疗机构总数 (Number of clients/hospitals)")
    parser.add_argument('--frac', type=float, default=1.0, help="每轮联邦通信中选择参与训练的客户端比例 (Fraction of clients to participate)")
    parser.add_argument('--local_ep', type=int, default=3, help="每个客户端本地训练的 Epoch 数 (Number of local epochs)")
    parser.add_argument('--local_bs', type=int, default=32, help="每个客户端本地训练的批次大小 (Local batch size)")
    parser.add_argument('--iid', type=bool, default=False, help="数据分布是否为独立同分布 (I.I.D.). 默认 False (Non-IID)")
    parser.add_argument('--alpha', type=float, default=1,help='Dirichlet distribution non-IID parameter. Smaller alpha means higher non-IID degree.')
    parser.add_argument('--seed', type=int, default=71, help='随机种子 (Random seed for reproducibility)')
    parser.add_argument('--seed_local', type=int, default=95,help='本地训练集划分种子。用于控制**全局训练集如何分配给各个客户端**，可修改此值来测试不同的本地数据分布。')

    # ===================================================================
    # 2. 数据集/环境 参数
    # ===================================================================
    parser.add_argument('--excel_path', type=str, default=r"D:\AWXY\研究生\小论文\数据图表\数据1+3.xlsx", help="Excel 文件路径 (Placeholder for data path)")
    parser.add_argument('--label_column', type=str, default="类型", help="标签列名 (Label column name)")
    # parser.add_argument('--feature_cols', type=list, default=["TV", "COV", "GI", "RVDI"],help="特征列名 (Feature column names)")
    parser.add_argument('--feature_cols', nargs='+', default=["TV", "COV", "GI", "RVDI"], help="...")
    # parser.add_argument('--val_size', type=float, default=0.15, help="集中式验证集比例 (Validation set size)")
    # parser.add_argument('--train_ratio', type=float, default=0.80,help="集中式训练集比例")
    parser.add_argument('--test_size', type=float, default=0.25, help="集中式测试集比例 (Test set size)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="使用的设备 ('cpu' or 'cuda')")

    # ===================================================================
    # 2.1 B 数据集单中心 RAN 二分类参数 (main_ran_B.py)
    # ===================================================================
    parser.add_argument('--ran_b_excel_path', type=str,
                        default=r"D:\projects\BRACETS\Data\00_数据总结\B_指标计算.xlsx",
                        help="B 数据集 Excel 文件路径")
    parser.add_argument('--ran_b_seed', type=int, default=62,
                        help="B 数据集单中心 RAN 训练集/测试集划分种子")
    parser.add_argument('--ran_b_test_size', type=float, default=0.25,
                        help="B 数据集单中心 RAN 测试集比例")

    # ===================================================================
    # 2.2 A/B 双客户端标准 FedAvg 参数 (main_fed_AB.py)
    # ===================================================================
    parser.add_argument('--fed_ab_a_excel_path', type=str,
                        default=r"D:\AWXY\研究生\小论文\数据图表\A_指标计算.xlsx",
                        help="标准 FedAvg 中 A 客户端数据集 Excel 文件路径")
    parser.add_argument('--fed_ab_b_excel_path', type=str,
                        default=r"D:\projects\BRACETS\Data\00_数据总结\B_指标计算.xlsx",
                        help="标准 FedAvg 中 B 客户端数据集 Excel 文件路径")
    parser.add_argument('--fed_ab_seed', type=int, default=62,
                        help="标准 FedAvg 中 A、B 数据集各自训练集/测试集划分种子")
    parser.add_argument('--fed_ab_test_size', type=float, default=0.25,
                        help="标准 FedAvg 中 A、B 数据集各自测试集比例")

    # ===================================================================
    # 2.3 A/B 双客户端轻量化联邦学习参数 (main_light_mix_fed_AB.py)
    # ===================================================================
    parser.add_argument('--light_mix_ab_a_excel_path', type=str,
                        default=r"D:\AWXY\研究生\小论文\数据图表\A_指标计算.xlsx",
                        help="A 客户端数据集 Excel 文件路径")
    parser.add_argument('--light_mix_ab_b_excel_path', type=str,
                        default=r"D:\projects\BRACETS\Data\00_数据总结\B_指标计算.xlsx",
                        help="B 客户端数据集 Excel 文件路径")
    parser.add_argument('--light_mix_ab_seed', type=int, default=62,
                        help="A、B 数据集各自训练集/测试集划分种子")
    parser.add_argument('--light_mix_ab_test_size', type=float, default=0.25,
                        help="A、B 数据集各自测试集比例")

    # ===================================================================
    # 3. RAN 模型结构参数 (参考你的原始设置)
    # ===================================================================
    parser.add_argument('--model_dim', type=int, default=64, help="每层通道数 (Feature dimension/Channel size in the model)")
    parser.add_argument('--num_blocks', type=int, default=2, help="注意力残差块数量 (Number of Attention Residual Blocks)")
    parser.add_argument('--hidden_mult', type=int, default=2, help="每块隐藏层扩大倍数 (Hidden layer multiplier)")
    parser.add_argument('--attn_reduction', type=int, default=4, help="注意力压缩比 (Attention reduction ratio)")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout 概率 (Dropout probability)")

    # ===================================================================
    # 3.1 FT-Transformer (FT-T) 模型结构参数 (已新增)
    # ===================================================================
    parser.add_argument('--ftt_d_token', type=int, default=192, help="FT-T: 每个特征 Token 的 embedding 维度 d ")
    parser.add_argument('--ftt_n_layers', type=int, default=3, help="FT-T: Transformer 层的数量 L ")
    parser.add_argument('--ftt_n_heads', type=int, default=8, help="FT-T: 多头注意力机制的头数 (Number of attention heads) ")
    parser.add_argument('--ftt_d_ffn_factor', type=float, default=1.3333, help="FT-T: 前馈神经网络 (FFN) 隐藏层维度的扩展倍数 (通常设为 4/3) ")
    parser.add_argument('--ftt_attention_dropout', type=float, default=0.2, help="FT-T: 注意力矩阵的 Dropout 概率 ")
    parser.add_argument('--ftt_ffn_dropout', type=float, default=0.1, help="FT-T: FFN 模块内部的 Dropout 概率 ")
    parser.add_argument('--ftt_residual_dropout', type=float, default=0.0, help="FT-T: 残差连接处的 Dropout 概率 ")

    # ===================================================================
    # 3.2 TabM 模型结构参数 (新增)
    # ===================================================================
    parser.add_argument('--tabm_k', type=int, default=32, help="TabM: 集成子模型数量 k (Number of ensemble members, 论文推荐 32)")
    parser.add_argument('--tabm_dim', type=int, default=128, help="TabM: 隐藏层维度 (Hidden layer dimension)")
    parser.add_argument('--tabm_layers', type=int, default=4, help="TabM: MLP 主干网络的层数 (Number of MLP layers)")
    parser.add_argument('--tabm_dropout', type=float, default=0.1, help="TabM: Dropout 概率")

    # ===================================================================
    # 4. 轻量化/通信优化参数 (更新)
    # ===================================================================
    # 4.1 本地加速参数 (对应 '冻结层')
    parser.add_argument('--freeze_blocks_count', type=int, default=0, help="客户端本地训练时，冻结 AttentionResidualBlock 的数量 (从块序列的开头开始计算)。0 表示不冻结。")

    # 4.2 通信优化参数 (对应 '结构稀疏 Delta-weight')
    parser.add_argument('--delta_weight_method', type=str, default='sparse',help="选择 Delta W 的压缩方法: 'sparse' (稀疏化), 'quant' (量化), 'none'. 默认 'sparse'")
    parser.add_argument('--delta_sparsity', type=float, default=0.985,help="Delta W 稀疏化率。例如 0.9 表示只上传 10% 的最大绝对值元素。")

    # 4.3 部署优化参数 (仅作记录，不参与训练流程)
    # parser.add_argument('--deployment_optimization', type=str, default='ptq', help="全局模型聚合后，用于部署的优化方法: 'ptq', 'pruning', 'none'.")
    # # 在 options.py 的第 4.3 节补充
    parser.add_argument('--do_pruning', type=bool, default=True, help="训练结束后是否进行全局剪枝")
    parser.add_argument('--pruning_percent', type=float, default=0.2, help="剪枝比例 (0.2 表示剪掉 20% 的权重)")

    parser.add_argument('--do_ptq', type=bool, default=False, help="训练结束后是否进行静态后量化 (PTQ)")

    # ===================================================================
    # 5. 优化器/训练超参数 (参考你的原始设置)
    # ===================================================================
    parser.add_argument('--lr', type=float, default=5e-4, help="学习率 (Learning rate)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2正则化 权重衰减 (Weight decay)")



    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 示例：如何获取参数
    args = args_parser()
    print("--- 联邦学习参数 ---")
    print(f"全局通信轮数: {args.epochs}")
    print(f"客户端总数: {args.num_users}")
    print(f"本地训练Epoch数: {args.local_ep}")
    print("\n--- 轻量化参数 ---")
    print(f"Delta权重方法: {args.delta_weight_method}")
    print(f"Delta稀疏率: {args.delta_sparsity}")
    print(f"剪枝比例: {args.pruning_percent}")
    print("\n--- FT-Transformer 参数 ---")
    print(f"Embedding 维度: {args.ftt_d_token}")
    print(f"Transformer 层数: {args.ftt_n_layers}")
    print(f"Attention 头数: {args.ftt_n_heads}")
    print("\n--- TabM 参数 ---")
    print(f"子模型数量 k: {args.tabm_k}")
    print(f"隐藏层维度: {args.tabm_dim}")
    print(f"MLP 层数: {args.tabm_layers}")
