"""
PGAC++ - Prototype-Guided Dual Attention Gating (物理引导的双重注意力门控)
基于丰度图的自适应通道+空间注意力模块（增强版）

核心改进：
1. F为主，A为辅：特征F的全局统计重新成为主角，A只做轻微修正
2. 轻量gate：gate接近1.0，只做细微re-weight（1±0.05左右）
3. 可学习的A贡献强度：alpha_a参数控制A的影响力，训练中自适应
4. 【新增】双重调制机制：
   - 通道分支：基于GAP的全局通道门控（原有机制）
   - 空间分支：直接利用丰度图A的空间位置信息进行逐像素调制
   - 目的：解决GAP丢失小目标信息的问题，提升小病灶分割能力

应用位置：
- 跳跃连接（Skip Connections）
- 在送入Up模块前对skip特征进行通道级+空间级门控

Version: 3.0 (Dual Attention: Channel + Spatial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGACPP(nn.Module):
    """
    PGAC++ - Prototype-Guided Dual Attention Gating (物理引导的双重注意力门控)
    
    核心思想：
    1. 用F的全局统计作为主要决策依据
    2. 用A的全局统计作为辅助调制
    3. gate ≈ 1.0 ± 0.1，不再是0~1的硬掐
    4. 【新增】双重调制：通道门控 × 空间门控
    
    Args:
        in_channels: 输入特征通道数
        num_proto: 原型/丰度图通道数
        reduction: MLP隐藏层缩减比例（默认4）
        mix_with_F: 是否融合F的全局统计（默认True）
        enable_spatial: 是否启用空间门控（默认True）
        epsilon_c: 通道门控调制幅度（默认0.1）
        epsilon_s: 空间门控调制幅度（默认0.1）
    """
    
    def __init__(self, 
                 in_channels: int, 
                 num_proto: int, 
                 reduction: int = 4,
                 mix_with_F: bool = True,
                 enable_spatial: bool = True,
                 epsilon_c: float = 0.1,
                 epsilon_s: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_proto = num_proto
        self.mix_with_F = mix_with_F
        self.enable_spatial = enable_spatial
        self.epsilon_c = epsilon_c
        self.epsilon_s = epsilon_s
        
        # 新增内容
        self.last_gate_channel = None      # (B,C,1,1)
        self.last_spatial_mask = None      # (B,1,H,W)  spatial_conv(A) logits
        self.last_gate_spatial = None      # (B,1,H,W)  mapped gate (1±eps)

        hidden = max(in_channels // reduction, 16)
        
        # ========== 通道分支（原有机制）==========
        # A分支：丰度图 -> 通道权重
        self.mlp_a = nn.Sequential(
            nn.Linear(num_proto, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels)
        )
        
        # F分支（可选）：特征 -> 通道权重
        if mix_with_F:
            self.mlp_f = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, in_channels)
            )
        
        # 控制A贡献强度的系数，初始很小（0.2）
        # 训练中会学到合适的值
        self.alpha_a = nn.Parameter(torch.tensor(0.2))
        
        # ========== 空间分支（新增）==========
        if enable_spatial:
            # 将K个丰度通道压缩为1个空间注意力图
            self.spatial_conv = nn.Conv2d(num_proto, 1, kernel_size=1, bias=True)
            # 初始化为较小的权重，使初始空间门控接近1.0
            nn.init.normal_(self.spatial_conv.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.spatial_conv.bias)
        
        print(f"    [PGAC++ Dual] C={in_channels}, K={num_proto}, alpha_a_init={0.2:.2f}, "
              f"mix_with_F={mix_with_F}, spatial={enable_spatial}, ε_c={epsilon_c:.2f}, ε_s={epsilon_s:.2f}")
    
    def forward(self, x_skip: torch.Tensor, A: torch.Tensor):
        """
        双重注意力门控前向传播
        
        Args:
            x_skip: (B, C, H, W) 跳跃连接特征
            A: (B, K, H, W) 对应尺度的丰度图
        
        Returns:
            gated_x: (B, C, H, W) 双重门控后的特征
        """
        B, C, H, W = x_skip.shape
        
        # ========== 通道分支：基于GAP的全局通道门控 ==========
        # Step 1: 全局池化
        a_g = A.mean(dim=(2, 3))        # (B, K) - 丰度全局统计
        
        # Step 2: 生成通道权重
        w_a = self.mlp_a(a_g)  # (B, C) - A的贡献
        
        if self.mix_with_F:
            f_g = x_skip.mean(dim=(2, 3))  # (B, C) - 特征全局统计
            w_f = self.mlp_f(f_g)          # (B, C) - F的贡献
            
            # F为主，A做修正
            logits = w_f + self.alpha_a * w_a
        else:
            logits = self.alpha_a * w_a
        
        # Step 3: 生成通道门控 g_channel (≈1.0±epsilon_c)
        g_sigmoid = torch.sigmoid(logits)  # (B, C) 范围 [0, 1]
        
        # 映射到 [1-epsilon_c, 1+epsilon_c]
        # gate_c = 1.0 + epsilon_c * (2*g - 1) = 1.0 + epsilon_c * (2*sigmoid - 1) ∈ [1±epsilon_c]
        gate_channel = 1.0 + self.epsilon_c * (2.0 * g_sigmoid - 1.0)  # (B, C)
        gate_channel = gate_channel.view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # ========== 空间分支：直接利用丰度图的空间位置信息 ==========
        if self.enable_spatial:
            # 将K个丰度通道压缩为1个空间mask
            spatial_mask = self.spatial_conv(A)  # (B, 1, H, W)
            
            # 生成空间门控 g_spatial (≈1.0±epsilon_s)
            # 使用sigmoid将值归一化到[0,1]，再映射到[1±epsilon_s]
            spatial_sigmoid = torch.sigmoid(spatial_mask)  # (B, 1, H, W)
            gate_spatial = 1.0 + self.epsilon_s * (2.0 * spatial_sigmoid - 1.0)  # (B, 1, H, W)
        else:
            gate_spatial = 1.0
        
        # ========== 双重调制融合 ==========
        # F_out = F * g_channel * g_spatial
        gated_x = x_skip * gate_channel * gate_spatial
        
        # 新增内容
        # ====== cache for visualization ======
        self.last_gate_channel = gate_channel.detach()

        if self.enable_spatial:
            self.last_spatial_mask = spatial_mask.detach()
            self.last_gate_spatial = gate_spatial.detach()
        else:
            self.last_spatial_mask = None
            self.last_gate_spatial = None


        return gated_x


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing PGAC++ Module (F-dominant version)")
    print("="*70)
    
    # 测试不同尺度的skip connections
    test_configs = [
        (2, 64, 256, 256, 4, "Skip 1: x1 -> up4"),
        (2, 128, 128, 128, 4, "Skip 2: x2 -> up3"),
        (2, 256, 64, 64, 4, "Skip 3: x3 -> up2"),
        (2, 512, 32, 32, 4, "Skip 4: x4 -> up1"),
        (2, 1024, 16, 16, 4, "Skip 5: x5 (bottleneck)"),
    ]
    
    for i, (B, C, H, W, K, desc) in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"Test {i+1}: {desc}")
        print(f"  Shape: (B={B}, C={C}, H={H}, W={W}), K={K}")
        print(f"{'='*70}")
        
        # 创建模块
        pgac = PGACPP(in_channels=C, num_proto=K, reduction=4, mix_with_F=True)
        
        # 创建输入
        x_skip = torch.randn(B, C, H, W)
        A = torch.randn(B, K, H, W)
        A = F.softmax(A, dim=1)  # 确保A是归一化的
        
        print(f"Input shapes:")
        print(f"  x_skip: {x_skip.shape}")
        print(f"  A: {A.shape}")
        
        # 前向传播
        with torch.no_grad():
            out = pgac(x_skip, A)
        
        print(f"Output shape: {out.shape}")
        assert out.shape == x_skip.shape, "Shape mismatch!"
        
        # 检查gate的值域（通道+空间）
        with torch.no_grad():
            a_g = A.mean(dim=(2, 3))
            w_a = pgac.mlp_a(a_g)
            f_g = x_skip.mean(dim=(2, 3))
            w_f = pgac.mlp_f(f_g)
            logits = w_f + pgac.alpha_a * w_a
            g_sigmoid = torch.sigmoid(logits)
            gate_channel = 1.0 + pgac.epsilon_c * (2.0 * g_sigmoid - 1.0)
            
            print(f"Channel Gate statistics:")
            print(f"  min: {gate_channel.min().item():.4f}")
            print(f"  max: {gate_channel.max().item():.4f}")
            print(f"  mean: {gate_channel.mean().item():.4f}")
            print(f"  std: {gate_channel.std().item():.4f}")
            
            if pgac.enable_spatial:
                spatial_mask = pgac.spatial_conv(A)
                spatial_sigmoid = torch.sigmoid(spatial_mask)
                gate_spatial = 1.0 + pgac.epsilon_s * (2.0 * spatial_sigmoid - 1.0)
                
                print(f"Spatial Gate statistics:")
                print(f"  min: {gate_spatial.min().item():.4f}")
                print(f"  max: {gate_spatial.max().item():.4f}")
                print(f"  mean: {gate_spatial.mean().item():.4f}")
                print(f"  std: {gate_spatial.std().item():.4f}")
        
        print(f"✓ Test {i+1} passed!")
    
    # 参数量统计
    print(f"\n{'='*70}")
    print("Parameter Statistics:")
    print(f"{'='*70}")
    
    for C in [64, 128, 256, 512, 1024]:
        pgac = PGACPP(C, num_proto=4, reduction=4, mix_with_F=True)
        total_params = sum(p.numel() for p in pgac.parameters())
        trainable_params = sum(p.numel() for p in pgac.parameters() if p.requires_grad)
        print(f"  C={C:4d}: {total_params:>6,} params (~{total_params/1000:.1f}K)")
    
    # 梯度测试
    print(f"\n{'='*70}")
    print("Gradient Test:")
    print(f"{'='*70}")
    pgac_test = PGACPP(128, num_proto=4)
    pgac_test.train()
    
    x_test = torch.randn(2, 128, 64, 64, requires_grad=True)
    A_test = F.softmax(torch.randn(2, 4, 64, 64), dim=1)
    
    out_test = pgac_test(x_test, A_test)
    loss = out_test.sum()
    loss.backward()
    
    print(f"✓ Backward pass successful")
    print(f"✓ Input gradient norm: {x_test.grad.norm().item():.6f}")
    print(f"✓ alpha_a grad: {pgac_test.alpha_a.grad.item():.6f}")
    print(f"✓ alpha_a value: {pgac_test.alpha_a.item():.6f}")
    
    # 测试不同alpha_a的影响
    print(f"\n{'='*70}")
    print("Testing alpha_a impact:")
    print(f"{'='*70}")
    
    x_demo = torch.randn(1, 64, 32, 32)
    A_demo = F.softmax(torch.randn(1, 4, 32, 32), dim=1)
    
    for alpha_val in [0.0, 0.1, 0.2, 0.5, 1.0]:
        pgac_demo = PGACPP(64, num_proto=4, mix_with_F=True)
        pgac_demo.alpha_a.data.fill_(alpha_val)
        
        with torch.no_grad():
            out_demo = pgac_demo(x_demo, A_demo)
            diff = (out_demo - x_demo).abs().mean()
        
        print(f"  alpha_a={alpha_val:.1f}: avg_abs_change={diff.item():.6f}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
