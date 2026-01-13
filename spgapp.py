"""
SPGA++ - Spectral Prototype-Guided Attention Plus Plus
基于丰度图的光谱原型引导注意力模块（增强版）

核心改进：
- 显式使用丰度图A参与注意力计算
- 通过concat[F, A]生成更精准的注意力权重

应用位置：
- 编码器深层：x3, x4, x5（在DoubleConv之后，CSSE之前）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPGAPP(nn.Module):
    """
    SPGA++ - Spectral Prototype-Guided Attention Plus Plus
    
    核心思想：
    - 将特征F和丰度图A拼接在通道维度
    - 通过轻量卷积生成注意力图
    - 对F进行残差增强
    
    Args:
        in_channels: 输入特征通道数
        num_proto: 原型/丰度图通道数
        reduction: 中间层缩减比例（默认4）
    """
    
    def __init__(self, in_channels: int, num_proto: int, reduction: int = 4):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_proto = num_proto
        
        mid = max(in_channels // reduction, 16)
        
        # 拼接[F, A]后的特征处理
        self.conv1 = nn.Conv2d(in_channels + num_proto, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        
        # 生成注意力权重
        self.conv2 = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.last_alpha = None        # (B,C,H,W)
        self.last_attn_map = None     # (B,1,H,W)  便于直接画热力图
        
        print(f"    [SPGA++] C={in_channels}, K={num_proto}, mid={mid}")
    
    def forward(self, F: torch.Tensor, A: torch.Tensor):
        """
        Args:
            F: (B, C, H, W) 特征图
            A: (B, K, H, W) 丰度图
        
        Returns:
            enhanced_F: (B, C, H, W) 增强后的特征
        """
        # ========== Step 1: 拼接特征和丰度 ==========
        Z = torch.cat([F, A], dim=1)  # (B, C+K, H, W)
        
        # ========== Step 2: 生成注意力权重 ==========
        x = torch.relu(self.bn1(self.conv1(Z)))      # (B, mid, H, W)
        alpha = torch.sigmoid(self.bn2(self.conv2(x)))  # (B, C, H, W)

        # 新增代码
        # ====== cache for visualization ======
        self.last_alpha = alpha.detach()
        self.last_attn_map = alpha.mean(dim=1, keepdim=True).detach()  # (B,1,H,W)
        
        self.last_attn_map = alpha.detach()
        
        # ========== Step 3: 残差增强 ==========
        return F + alpha * F


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing SPGA++ Module")
    print("="*70)
    
    # 测试不同尺度
    test_configs = [
        (2, 256, 64, 64, 4, "x3 (编码器 down2)"),
        (2, 512, 32, 32, 4, "x4 (编码器 down3)"),
        (2, 1024, 16, 16, 4, "x5 (编码器 down4/bottleneck)"),
    ]
    
    for i, (B, C, H, W, K, desc) in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"Test {i+1}: {desc}")
        print(f"  Shape: (B={B}, C={C}, H={H}, W={W}), K={K}")
        print(f"{'='*70}")
        
        # 创建模块
        spga = SPGAPP(in_channels=C, num_proto=K, reduction=4)
        
        # 创建输入
        F = torch.randn(B, C, H, W)
        A = torch.randn(B, K, H, W)
        A = torch.softmax(A, dim=1)  # 确保A是归一化的
        
        print(f"Input shapes:")
        print(f"  F: {F.shape}")
        print(f"  A: {A.shape}")
        
        # 前向传播
        with torch.no_grad():
            out = spga(F, A)
        
        print(f"Output shape: {out.shape}")
        assert out.shape == F.shape, "Shape mismatch!"
        
        print(f"✓ Test {i+1} passed!")
    
    # 参数量统计
    print(f"\n{'='*70}")
    print("Parameter Statistics:")
    print(f"{'='*70}")
    
    for C in [256, 512, 1024]:
        spga = SPGAPP(C, num_proto=4, reduction=4)
        total_params = sum(p.numel() for p in spga.parameters())
        trainable_params = sum(p.numel() for p in spga.parameters() if p.requires_grad)
        print(f"  C={C:4d}: {total_params:>6,} params (~{total_params/1000:.1f}K)")
    
    # 梯度测试
    print(f"\n{'='*70}")
    print("Gradient Test:")
    print(f"{'='*70}")
    spga_test = SPGAPP(256, num_proto=4)
    spga_test.train()
    
    F_test = torch.randn(2, 256, 64, 64, requires_grad=True)
    A_test = torch.softmax(torch.randn(2, 4, 64, 64), dim=1)
    
    out_test = spga_test(F_test, A_test)
    loss = out_test.sum()
    loss.backward()
    
    print(f"✓ Backward pass successful")
    print(f"✓ Input gradient norm: {F_test.grad.norm().item():.6f}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
