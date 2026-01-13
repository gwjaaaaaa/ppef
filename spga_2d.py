"""
SPGA - Spectral Prototype-Guided Adaptive Attention (2D版本)
光谱原型引导的自适应注意力模块

迁移自nnUNet v2的3D实现，适配到2D高光谱图像分割

核心创新：
1. 可学习的原型库 - 学习K=4个代表性通道模式（对应3D中的光谱模式）
2. 原型匹配机制 - 通过特征与原型的相似度生成动态注意力
3. 通道-空间解耦重耦合 - 先分离再自适应融合
4. 可解释性 - 可可视化学到的原型和激活图

关键转换（3D → 2D）：
- 输入: (B, C, H, W, D) → (B, C, H, W)
- 光谱维度D → 通道维度C（60通道包含光谱信息）
- Conv3d → Conv2d
- 原型学习：从光谱空间(K, D)适配到通道空间(K, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelPrototypeBank(nn.Module):
    """
    通道原型库 (2D适配版)
    
    创新点：学习一组可学习的通道原型向量，代表不同的通道响应模式
    
    3D→2D转换：
    - 原型维度：(K, spectral_dim=40) → (K, 通道数C)
    - 每个stage学习自己的原型（256/512/1024通道）
    """
    def __init__(self, channels, num_prototypes=4):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        
        # 可学习的原型矩阵: (num_prototypes, channels)
        # 每个原型代表一种通道响应模式
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, channels) * 0.01
        )
        
        # 归一化
        self.layer_norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        """
        输入: x (B, C, H, W)
        输出: 
            - prototype_codes: (B, K, H, W) 原型激活图
            - prototypes_norm: (K, C) 归一化后的原型
        """
        B, C, H, W = x.shape
        
        # 归一化原型
        prototypes_norm = self.layer_norm(self.prototypes)  # (K, C)
        
        # 提取输入的通道特征: (B, C, H, W) → (B, H, W, C)
        # 每个空间位置有一个C维通道向量
        channel_features = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # 计算与原型的相似度
        # Reshape: (B, H*W, C)
        channel_flat = channel_features.reshape(B, H*W, C)  # (B, H*W, C)
        
        # 归一化用于余弦相似度
        channel_flat_norm = F.normalize(channel_flat, p=2, dim=-1)  # (B, H*W, C)
        prototypes_unit = F.normalize(prototypes_norm, p=2, dim=-1)  # (K, C)
        
        # 计算相似度: (B, H*W, C) x (K, C)^T = (B, H*W, K)
        similarity = torch.matmul(channel_flat_norm, prototypes_unit.T)  # (B, H*W, K)
        
        # Softmax归一化，得到原型激活
        prototype_activation = F.softmax(similarity * 10.0, dim=-1)  # (B, H*W, K) 温度=10
        
        # Reshape回空间维度
        prototype_codes = prototype_activation.permute(0, 2, 1).view(B, self.num_prototypes, H, W)  # (B, K, H, W)
        
        return prototype_codes, prototypes_norm


class ChannelSpatialDecoupling(nn.Module):
    """
    通道-空间解耦模块 (2D适配版 - 参数优化版)
    
    创新点：显式地将特征分解为通道分量和空间分量，然后再进行自适应融合
    
    参数优化：使用深度可分离卷积大幅减少参数量
    - 标准Conv2d(C, C, 3): 9C²参数
    - Depthwise Conv2d(C, C, 3, groups=C) + Pointwise Conv2d(C, C, 1): 9C + C² 参数
    - 减少约9倍参数量（对于大通道数）
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 通道分支：提取通道间关系（类似3D的光谱分支）
        # 使用1x1卷积（通道间交互）
        self.channel_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # 空间分支：提取空间特征
        # 使用深度可分离卷积（大幅减少参数）
        self.spatial_branch = nn.Sequential(
            # Depthwise卷积（空间维度处理）
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Pointwise卷积（通道维度融合）
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # 融合门控：动态决定通道和空间的权重
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        输入: x (B, C, H, W)
        输出: decoupled_features (B, C, H, W)
        """
        # 通道特征
        channel_features = self.channel_branch(x)  # (B, C, H, W)
        
        # 空间特征
        spatial_features = self.spatial_branch(x)  # (B, C, H, W)
        
        # 自适应融合
        concat_features = torch.cat([channel_features, spatial_features], dim=1)  # (B, 2C, H, W)
        gate = self.fusion_gate(concat_features)  # (B, C, 1, 1)
        
        # 门控融合：gate控制通道和空间的比例
        fused_features = gate * channel_features + (1 - gate) * spatial_features
        
        return fused_features


class PrototypeGuidedAttention(nn.Module):
    """
    原型引导的注意力生成 (2D适配版)
    
    创新点：基于原型匹配结果生成空间注意力和通道注意力
    
    3D→2D转换：
    - 输入维度：(B, K, H, W, 1) → (B, K, H, W)
    - 输出维度：(B, C, 1, 1, 1) → (B, C, 1, 1)
    """
    def __init__(self, channels, num_prototypes=4):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        
        # 原型到空间注意力的映射
        self.spatial_attention_gen = nn.Sequential(
            nn.Conv2d(num_prototypes, num_prototypes // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_prototypes // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 原型到通道注意力的映射
        self.channel_attention_gen = nn.Sequential(
            nn.Linear(num_prototypes, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, prototype_codes):
        """
        输入:
            - prototype_codes: (B, K, H, W) 原型激活图
        输出:
            - spatial_attn: (B, 1, H, W) 空间注意力
            - channel_attn: (B, C, 1, 1) 通道注意力
        """
        B, K, H, W = prototype_codes.shape
        
        # 生成空间注意力
        spatial_attn = self.spatial_attention_gen(prototype_codes)  # (B, 1, H, W)
        
        # 生成通道注意力
        # 全局原型激活: (B, K, H, W) -> (B, K)
        global_prototype_activation = prototype_codes.mean(dim=[2, 3])  # (B, K)
        channel_attn = self.channel_attention_gen(global_prototype_activation)  # (B, C)
        channel_attn = channel_attn.view(B, self.channels, 1, 1)  # (B, C, 1, 1)
        
        return spatial_attn, channel_attn


class SPGAModule2D(nn.Module):
    """
    完整的SPGA模块 (2D适配版 - 完整版，4个原型)
    
    创新点总结：
    1. 通道原型学习 - 自动从数据中学习代表性通道模式（对应3D中的光谱模式）
    2. 原型匹配注意力 - 基于原型相似度生成注意力
    3. 通道-空间解耦 - 显式建模两种特征的交互
    4. 可解释性 - 可以可视化学到的原型和激活图
    
    输入: (B, C, H, W)
    输出: (B, C, H, W)  # 尺度保持不变
    
    应用位置（对应nnUNet的stage 2,3,4）：
    - down2后: 256通道 (256x320分辨率)
    - down3后: 512通道 (128x160分辨率)  
    - down4后: 1024通道 (64x80分辨率)
    """
    def __init__(self, 
                 channels, 
                 num_prototypes=4,
                 use_residual=True,
                 dropout_rate=0.1):
        super().__init__()
        
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        
        # 1. 通道原型库
        self.prototype_bank = ChannelPrototypeBank(channels, num_prototypes)
        
        # Dropout层（防止过拟合）
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        # 2. 通道-空间解耦
        self.decoupling = ChannelSpatialDecoupling(channels)
        
        # 3. 原型引导注意力
        self.attention_gen = PrototypeGuidedAttention(channels, num_prototypes)
        
        # 4. 特征增强
        self.enhancement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        print(f"  [SPGA-2D] Initialized:")
        print(f"    - Channels: {channels}")
        print(f"    - Num prototypes: {num_prototypes}")
        print(f"    - Learnable prototype parameters: {num_prototypes * channels}")
        print(f"    - Use residual: {use_residual}")
    
    def forward(self, x):
        """
        输入: x (B, C, H, W)
        输出: enhanced_x (B, C, H, W)
        
        处理流程：
        1. 原型匹配 -> 得到原型激活图
        2. 解耦通道和空间特征
        3. 基于原型生成注意力
        4. 应用注意力增强特征
        """
        identity = x  # 残差连接
        
        B, C, H, W = x.shape
        
        # Step 1: 通道原型匹配
        prototype_codes, prototypes_norm = self.prototype_bank(x)
        # prototype_codes: (B, K, H, W)
        # prototypes_norm: (K, C)
        
        # Step 2: 通道-空间解耦
        decoupled_features = self.decoupling(x)  # (B, C, H, W)
        
        # Step 3: 生成注意力
        spatial_attn, channel_attn = self.attention_gen(prototype_codes)
        # spatial_attn: (B, 1, H, W)
        # channel_attn: (B, C, 1, 1)
        
        # Step 4: 应用注意力
        # 先应用通道注意力
        enhanced = decoupled_features * channel_attn  # (B, C, H, W)
        
        # 再应用空间注意力
        enhanced = enhanced * spatial_attn  # (B, C, H, W)
        
        # Step 5: 特征增强
        enhanced = self.enhancement(enhanced)
        
        # Step 5.5: Dropout（防止过拟合）
        if self.dropout is not None and self.training:
            enhanced = self.dropout(enhanced)
        
        # Step 6: 残差连接
        if self.use_residual:
            output = enhanced + identity
        else:
            output = enhanced
        
        return output
    
    def get_prototype_visualization(self):
        """
        获取学到的原型，用于可视化和分析
        返回: (num_prototypes, channels)
        """
        return self.prototype_bank.prototypes.detach().cpu()


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing SPGA 2D Module (完整版，4个原型)")
    print("="*70)
    
    # 测试参数（模拟实际使用场景）
    test_configs = [
        (2, 256, 256, 320),   # down2后: 256通道
        (2, 512, 128, 160),   # down3后: 512通道
        (2, 1024, 64, 80),    # down4后: 1024通道
    ]
    
    for i, (B, C, H, W) in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"Test {i+1}: (B={B}, C={C}, H={H}, W={W})")
        print(f"{'='*70}")
        
        # 创建模块
        spga = SPGAModule2D(channels=C, num_prototypes=4)
        
        # 创建输入
        x = torch.randn(B, C, H, W)
        print(f"Input shape: {x.shape}")
        
        # 前向传播
        with torch.no_grad():
            out = spga(x)
        
        print(f"Output shape: {out.shape}")
        assert out.shape == x.shape, "Shape mismatch!"
        
        # 获取原型
        prototypes = spga.get_prototype_visualization()
        print(f"Prototypes shape: {prototypes.shape}")
        print(f"Prototype norm: {prototypes.norm(dim=1)}")
        print(f"✓ Test {i+1} passed!")
    
    # 参数量统计
    print(f"\n{'='*70}")
    print("Parameter Statistics (for 256 channels):")
    print(f"{'='*70}")
    spga_256 = SPGAModule2D(256, num_prototypes=4)
    total_params = sum(p.numel() for p in spga_256.parameters())
    trainable_params = sum(p.numel() for p in spga_256.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Memory (FP32): ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)

