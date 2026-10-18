# models/Nets.py

import torch
import torch.nn as nn
from collections import OrderedDict
# 【新增】导入量化工具
# 修改后的写法
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional


# =====================================================
# ========= 1. 核心模块定义 (ResidualMLPBlock) ========
# =====================================================

class ResidualMLPBlock(nn.Module):
    """
    残差多层感知机块（Residual MLP Block）
    """

    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()

        # 【修改】拆解 Sequential 以便进行 Linear+BN 融合
        self.fc1 = nn.Linear(dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(drop)

        self.fc2 = nn.Linear(hidden, dim)
        self.bn2 = nn.BatchNorm1d(dim)

        self.act_out = nn.GELU()

        # 【新增】量化加法算子 (替换 + 号)
        self.f_add = FloatFunctional()

    def forward(self, x):
        identity = x

        # 拆解后的前向传播
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.bn2(out)

        # 实现 x -> net(x) + x 的残差连接
        # 【修改】使用 f_add.add 替换 +
        out = self.f_add.add(out, identity)

        return self.act_out(out)

    def fuse_model(self):
        # 【新增】定义融合规则：Linear + BN
        torch.quantization.fuse_modules(self, [['fc1', 'bn1'], ['fc2', 'bn2']], inplace=True)


# =====================================================
# ========== 2. 注意力模块定义 (ChannelAttention) ==========
# =====================================================

class ChannelAttention(nn.Module):
    """
    通道注意力模块 (Squeeze-and-Excitation 变体)
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        # 确保隐藏层维度至少为 1
        h = max(1, dim // reduction)

        # 【修改】拆解 Sequential
        self.fc1 = nn.Linear(dim, h)
        self.act1 = nn.ReLU()  # ReLU 对量化更友好，通常可融合
        self.fc2 = nn.Linear(h, dim)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x

    def fuse_model(self):
        # 【新增】融合 Linear + ReLU (如果有的话)
        torch.quantization.fuse_modules(self, [['fc1', 'act1']], inplace=True)


# =====================================================
# ========= 3. 注意力残差块 (AttentionResidualBlock) ========
# =====================================================

class AttentionResidualBlock(nn.Module):
    """
    注意力残差块 (Trunk + Attention Gate)
    """

    def __init__(self, dim, hidden, attn_reduction=4, drop=0.0):
        super().__init__()
        # Trunk path: 使用 ResidualMLPBlock
        self.trunk = ResidualMLPBlock(dim, hidden, drop)
        # Attention path: 使用 ChannelAttention 得到注意力掩码
        self.attn = ChannelAttention(dim, attn_reduction)

        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.GELU()

        # 【新增】量化乘法和加法算子
        self.f_mul = FloatFunctional()
        self.f_add = FloatFunctional()

    def forward(self, x):
        # Trunk output: T
        t = self.trunk(x)
        # Attention mask: M, 范围在 (0, 1)
        m = self.attn(x)

        # Attention Residual Connection: H = (1 + M) * T + X
        # 转换为量化友好的等价逻辑: H = (M * T) + T + X

        # 1. M * T
        m_t = self.f_mul.mul(m, t)

        # 2. (M * T) + T
        attended_t = self.f_add.add(m_t, t)

        # 3. + X (Residual)
        h = self.f_add.add(attended_t, x)

        return self.act(self.norm(h))

    def fuse_model(self):
        # 【新增】递归调用子模块的融合
        self.trunk.fuse_model()
        self.attn.fuse_model()


# =====================================================
# ========== 4. 最终网络结构 (RANTabular) ==========
# =====================================================

class RANTabular(nn.Module):
    """
    残差注意力网络 RANTabular (适用于表格数据)
    """

    def __init__(self, in_dim, num_classes, model_dim=64, num_blocks=4,
                 hidden_mult=2, attn_reduction=4, dropout=0.1):
        super().__init__()

        # 【新增】量化桩 (Stub)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # 1. 输入投影层 (Projection) - 【修改】拆解
        self.proj_fc = nn.Linear(in_dim, model_dim)
        self.proj_bn = nn.BatchNorm1d(model_dim)
        self.proj_act = nn.GELU()

        # 2. 注意力残差块序列 (Blocks)
        hidden = model_dim * hidden_mult
        # 【修改】使用 ModuleList 以便遍历
        self.blocks = nn.ModuleList([
            AttentionResidualBlock(model_dim, hidden, attn_reduction, drop=dropout)
            for _ in range(num_blocks)
        ])

        # 3. 分类头 (Classification Head) - 【修改】拆解
        self.head_bn = nn.BatchNorm1d(model_dim)
        self.head_act = nn.GELU()
        self.head_drop = nn.Dropout(dropout)
        self.head_fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # 【新增】开启量化
        x = self.quant(x)

        # 1. Projection
        x = self.proj_fc(x)
        x = self.proj_bn(x)
        x = self.proj_act(x)

        # 2. Blocks
        for block in self.blocks:
            x = block(x)

        # 3. Head
        x = self.head_bn(x)
        x = self.head_act(x)
        x = self.head_drop(x)
        x = self.head_fc(x)

        # 【新增】结束量化
        x = self.dequant(x)

        return x

    def fuse_model(self):
        """
        执行算子融合：将 Linear 和 BatchNorm 融合成一层
        """
        # 1. 融合 Projection (Linear + BN)
        torch.quantization.fuse_modules(self, [['proj_fc', 'proj_bn']], inplace=True)

        # 2. 递归融合 Blocks
        for block in self.blocks:
            block.fuse_model()

        # Head 部分通常保持原样