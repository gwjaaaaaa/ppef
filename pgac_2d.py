"""
PGAC - Prototype-Guided Adaptive Channel Attention (2D版本)
原型引导的自适应通道注意力

迁移自nnUNet v2的3D实现，适配到2D高光谱图像分割

核心创新：
1. 原型引导通道注意力 - 不用SE-Net的黑盒全局池化，而是用原型引导
2. 自适应融合 - 结合原型知识和局部统计信息
3. 轻量级设计 - 参数量远小于SE-Net
4. 可解释性 - 可以可视化哪个原型被激活

应用位置：
- 跳跃连接（skip connections）：在编码器特征传递到解码器之前进行通道筛选

关键转换（3D → 2D）：
- 输入: (B, C, H, W, D) → (B, C, H, W)
- Conv3d → Conv2d
- 原型维度：(K, C) 保持不变
- 在通道空间学习原型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PGACModule2D(nn.Module):
    """
    Prototype-Guided Adaptive Channel Attention (2D适配版)
    
    替代SE-Net，用原型引导通道注意力
    
    工作流程：
    1. 学习K个通道原型 P ∈ R^(K×C)
    2. 计算skip特征与原型的相似度
    3. 基于相似度生成原型引导的通道权重
    4. 结合局部通道统计，自适应融合
    5. 应用通道权重到skip特征
    
    输入: (B, C, H, W) skip特征（来自编码器）
    输出: (B, C, H, W) 通道注意力加权后的特征
    """
    
    def __init__(
        self,
        channels: int,
        num_prototypes: int = 4,
        reduction: int = 8,
        use_local_stats: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            channels: 特征通道数
            num_prototypes: 原型数量（默认4，与SPGA保持一致）
            reduction: 局部统计网络的压缩比（默认8）
            use_local_stats: 是否使用局部统计信息（默认True）
            dropout_rate: Dropout概率（默认0.1）
        """
        super().__init__()
        
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.use_local_stats = use_local_stats
        self.dropout_rate = dropout_rate
        
        # === 原型存储（可学习参数）===
        # 在通道空间学习原型
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, channels) * 0.01
        )
        
        # === 局部通道统计网络（轻量）===
        if use_local_stats:
            mid_channels = max(channels // reduction, 8)
            self.local_net = nn.Sequential(
                nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(mid_channels),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        
        # === 自适应融合系数（可学习）===
        # α控制原型引导 vs 局部统计的权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始0.5，训练中学习
        
        # === Dropout层（防止过拟合）===
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    [PGAC-2D] C={channels}, K={num_prototypes}, R={reduction}")
        print(f"      ├─ Prototype params: {self.prototypes.numel():,}")
        if use_local_stats:
            local_params = sum(p.numel() for p in self.local_net.parameters())
            print(f"      ├─ Local net params: {local_params:,}")
        print(f"      └─ Total params: {total_params:,} (~{total_params/1000:.1f}K)")
    
    def set_prototypes(self, prototypes: torch.Tensor):
        """
        从SPGA模块注入原型（可选）
        
        Args:
            prototypes: (K, C) 从SPGA学到的原型
        """
        with torch.no_grad():
            self.prototypes.copy_(prototypes)
    
    def compute_prototype_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算特征与原型的相似度
        
        Args:
            x: (B, C, H, W) skip特征
        Returns:
            similarity: (B, K) 与每个原型的相似度
        """
        # 全局平均池化
        x_gap = x.mean(dim=[2, 3])  # (B, C)
        
        # 归一化
        x_norm = F.normalize(x_gap, dim=1)  # (B, C)
        proto_norm = F.normalize(self.prototypes, dim=1)  # (K, C)
        
        # 计算余弦相似度
        similarity = torch.matmul(x_norm, proto_norm.T)  # (B, K)
        
        # Softmax归一化（哪个原型最相关）
        similarity = F.softmax(similarity, dim=1)  # (B, K)
        
        return similarity
    
    def prototype_guided_weights(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        基于原型相似度生成通道权重
        
        Args:
            similarity: (B, K) 原型相似度
        Returns:
            weights: (B, C) 原型引导的通道权重
        """
        # 加权组合原型 → 得到"理想的通道权重分布"
        # weights = ∑(similarity_k · prototype_k)
        weights = torch.matmul(similarity, self.prototypes)  # (B, C)
        
        # Sigmoid归一化到[0, 1]
        weights = torch.sigmoid(weights)
        
        return weights
    
    def forward(self, x: torch.Tensor, external_prototypes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, C, H, W) skip特征
            external_prototypes: (K, C) 可选的外部原型（来自SPGA），如果提供则优先使用
        Returns:
            out: (B, C, H, W) 通道注意力加权后的特征
        """
        B, C, H, W = x.shape
        
        # === Step 1: 选择原型源 ===
        if external_prototypes is not None:
            # 使用SPGA提供的原型（动态更新）
            prototypes_to_use = external_prototypes
        else:
            # 使用内部可学习原型
            prototypes_to_use = self.prototypes
        
        # === Step 2: 计算与原型的相似度 ===
        # 全局平均池化
        x_gap = x.mean(dim=[2, 3])  # (B, C)
        
        # 归一化
        x_norm = F.normalize(x_gap, dim=1)  # (B, C)
        proto_norm = F.normalize(prototypes_to_use, dim=1)  # (K, C)
        
        # 计算余弦相似度
        similarity = torch.matmul(x_norm, proto_norm.T)  # (B, K)
        similarity = F.softmax(similarity, dim=1)  # (B, K)
        
        # === Step 3: 原型引导的通道权重 ===
        w_proto = torch.matmul(similarity, prototypes_to_use)  # (B, C)
        w_proto = torch.sigmoid(w_proto)
        
        # === Step 4: 局部通道统计 ===
        if self.use_local_stats:
            w_local = self.local_net(x).mean(dim=[2, 3])  # (B, C)
        else:
            w_local = torch.ones_like(w_proto)
        
        # === Step 5: 自适应融合 ===
        # α ∈ [0, 1]，控制原型 vs 局部的权重
        alpha = torch.sigmoid(self.alpha)  # 限制在[0, 1]
        w_final = alpha * w_proto + (1 - alpha) * w_local  # (B, C)
        
        # === Step 6: 应用通道权重 ===
        w_final = w_final.view(B, C, 1, 1)  # (B, C, 1, 1)
        out = x * w_final
        
        # === Step 6.5: Dropout（防止过拟合）===
        if self.dropout is not None and self.training:
            out = self.dropout(out)
        
        return out


def build_pgac_module(
    channels: int,
    num_prototypes: int = 4,
    reduction: int = 8,
    use_local_stats: bool = True
) -> PGACModule2D:
    """
    构建PGAC模块
    
    Args:
        channels: 通道数
        num_prototypes: 原型数量（与SPGA保持一致）
        reduction: 局部网络压缩比
        use_local_stats: 是否使用局部统计
    Returns:
        PGAC模块实例
    """
    return PGACModule2D(
        channels=channels,
        num_prototypes=num_prototypes,
        reduction=reduction,
        use_local_stats=use_local_stats
    )


# 测试代码
if __name__ == "__main__":
    print("="*80)
    print("Testing PGAC 2D Module")
    print("="*80)
    
    # 测试参数（模拟跳跃连接的不同通道数）
    test_configs = [
        (2, 64, 512, 640),     # skip from in_conv
        (2, 128, 256, 320),    # skip from down1
        (2, 256, 128, 160),    # skip from down2
        (2, 512, 64, 80),      # skip from down3
        (2, 1024, 32, 40),     # skip from down4
    ]
    
    for i, (B, C, H, W) in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"Test {i+1}: (B={B}, C={C}, H={H}, W={W})")
        print(f"{'='*80}")
        
        # 创建模块
        pgac = build_pgac_module(
            channels=C,
            num_prototypes=4,
            reduction=8,
            use_local_stats=True
        )
        
        # 模拟SPGA原型（可选）
        mock_prototypes = torch.randn(4, C)
        
        # 创建输入
        x = torch.randn(B, C, H, W)
        print(f"Input shape: {x.shape}")
        
        # 前向传播（使用内部原型）
        with torch.no_grad():
            out1 = pgac(x)
        print(f"Output shape (internal prototypes): {out1.shape}")
        assert out1.shape == x.shape, "Shape mismatch!"
        
        # 前向传播（使用外部原型）
        with torch.no_grad():
            out2 = pgac(x, external_prototypes=mock_prototypes)
        print(f"Output shape (external prototypes): {out2.shape}")
        assert out2.shape == x.shape, "Shape mismatch!"
        
        print(f"Alpha value: {torch.sigmoid(pgac.alpha).item():.4f}")
        print(f"✓ Test {i+1} passed!")
    
    # 梯度测试
    print(f"\n{'='*80}")
    print("Gradient Test:")
    print(f"{'='*80}")
    pgac_test = build_pgac_module(128, 4)
    pgac_test.train()
    x_test = torch.randn(2, 128, 256, 256, requires_grad=True)
    out_test = pgac_test(x_test)
    loss = out_test.sum()
    loss.backward()
    
    print(f"✓ Backward pass successful")
    print(f"✓ Prototype grad norm: {pgac_test.prototypes.grad.norm().item():.6f}")
    print(f"✓ Alpha grad: {pgac_test.alpha.grad.item():.6f}")
    
    # 参数量统计
    print(f"\n{'='*80}")
    print("Parameter Statistics:")
    print(f"{'='*80}")
    
    for C in [64, 128, 256, 512, 1024]:
        pgac = build_pgac_module(C, 4, 8)
        total_params = sum(p.numel() for p in pgac.parameters())
        print(f"  C={C:4d}: {total_params:>8,} params (~{total_params/1000:.1f}K)")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

