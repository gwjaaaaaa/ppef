"""
UNet_PPEF++ - Enhanced PPEF Framework
完整增强PPEF框架

集成模块：
1. SpectralUnmixingHead - 光谱解混头
2. CSSE - 通道-空间-光谱专家模块
3. PGAC++ - 基于丰度图的通道门控
4. SPGA++ - 基于丰度图的增强注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从model.py导入基础模块
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import DoubleConv, Down, Up, OutConv

# 导入PPEF++模块
try:
    from ppef_modules import SpectralUnmixingHead, CSSE, PGACPP, SPGAPP
    PPEF_AVAILABLE = True
except ImportError:
    PPEF_AVAILABLE = False
    print("Warning: PPEF++ modules not available.")


class UNet_PPEFPP(nn.Module):
    """
    UNet with PPEF++ Framework (完整增强PPEF框架)
    集成SpectralUnmixingHead + CSSE + PGAC++ + SPGA++
    
    核心改进：
    1. 光谱解混头（Unmixing Head）- 在down2后生成丰度图A2
    2. CSSE模块 - 4专家系统（替代DSR，编码/解码共用）
    3. PGAC++ - 基于丰度图的通道门控（替代PGAC）
    4. SPGA++ - 基于丰度图的增强注意力（替代SPGA）
    
    完整的知识流动路径：
    
    编码器（Unmixing + SPGA++ + CSSE）:
    - x1 = in_conv(x)                        # (B, 64, 256, 256)
    - x2 = down1(x1)                         # (B, 128, 128, 128)
    - x3 = down2(x2)                         # (B, 256, 64, 64)
    - A2, X2_hat, X2_down = unmix_head(x3, x)  # 生成丰度图
    - x2 = CSSE_enc1(x2, A1)                 # A1从A2插值得到
    - x3 = SPGA++(x3, A2) -> CSSE_enc2(x3, A2)
    - x4 = down3(x3) -> SPGA++(x4, A3) -> CSSE_enc3(x4, A3)
    - x5 = down4(x4)
    
    跳跃连接（PGAC++）:
    - s1 = x1               # 浅层不用PGAC++，避免A2过度控制
    - s2 = x2               # 浅层不用PGAC++
    - s3 = PGAC++(x3, A2)   # 只在深层使用PGAC++
    - s4 = PGAC++(x4, A3)
    - s5 = PGAC++(x5, A4)
    
    解码器（CSSE）:
    - d4 = up1(x5, s4) -> CSSE_dec1(d4, A3)
    - d3 = up2(d4, s3) -> CSSE_dec2(d3, A2)
    - d2 = up3(d3, s2) -> CSSE_dec3(d2, A1)
    - d1 = up4(d2, s1)
    - out = out_conv(d1)
    
    损失函数：
    - L_seg: 分割损失（Dice + BCE）
    - L_recon: 光谱重建损失 MSE(X2_hat, X2_down)
    - L_smooth: 丰度图平滑损失
    - L_div: 原型多样性损失
    - L_entropy: 丰度熵损失
    """
    
    def __init__(self, 
                 in_channels=40, 
                 out_channels=1,
                 num_prototypes=4,
                 dropout_rate=0.1,
                 use_spgapp=True,
                 use_csse=True,
                 use_pgacpp=True):
        """
        Args:
            in_channels: 输入通道数（60个光谱通道）
            out_channels: 输出通道数（单通道二分类用1）
            num_prototypes: 原型数量（默认4）
            dropout_rate: Dropout概率（默认0.1）
            use_spgapp: 是否使用SPGA++模块
            use_csse: 是否使用CSSE模块
            use_pgacpp: 是否使用PGAC++模块
        """
        super(UNet_PPEFPP, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_prototypes = num_prototypes
        self.num_bands = in_channels  # 高光谱波段数
        self.dropout_rate = dropout_rate
        
        self.use_spgapp = use_spgapp and PPEF_AVAILABLE
        self.use_csse = use_csse and PPEF_AVAILABLE
        self.use_pgacpp = use_pgacpp and PPEF_AVAILABLE
        
        if not PPEF_AVAILABLE:
            print("Warning: PPEF++ modules requested but not available. Falling back to standard UNet.")
        
        # ========== 编码器 ==========
        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # ========== PPEF++ 模块初始化 ==========
        if any([self.use_spgapp, self.use_csse, self.use_pgacpp]):
            print("\n" + "="*70)
            print("Initializing UNet with PPEF++ Framework:")
            print("="*70)
        
        # 1. 光谱解混头（在down2后生成A2）
        print("\n[Spectral Unmixing Head]")
        self.unmix_head = SpectralUnmixingHead(
            feat_channels=256,  # down2输出通道数
            num_bands=self.num_bands,
            num_proto=num_prototypes
        )
        
        # 2. SPGA++模块（编码器）
        if self.use_spgapp:
            print("\n[SPGA++ - Abundance-Guided Attention]")
            self.spga3 = SPGAPP(in_channels=256, num_proto=num_prototypes)  # x3 (down2后)
            self.spga4 = SPGAPP(in_channels=512, num_proto=num_prototypes)  # x4 (down3后)
        
        # 3. CSSE模块（编码器 + 解码器）
        # if self.use_csse:
        #     print("\n[CSSE - Channel-Spatial-Spectrum Expert]")
        #     print("  Encoder:")
        #     self.csse_enc1 = CSSE(channels=128, num_proto=num_prototypes, enable_expert4=False)  # x2
        #     self.csse_enc2 = CSSE(channels=256, num_proto=num_prototypes, enable_expert4=False)  # x3
        #     self.csse_enc3 = CSSE(channels=512, num_proto=num_prototypes, enable_expert4=False)  # x4
            
        #     print("  Decoder:")
        #     self.csse_dec1 = CSSE(channels=512, num_proto=num_prototypes, enable_expert4=False)  # up1 output
        #     self.csse_dec2 = CSSE(channels=256, num_proto=num_prototypes, enable_expert4=False)  # up2 output
        #     self.csse_dec3 = CSSE(channels=128, num_proto=num_prototypes, enable_expert4=False)  # up3 output

        # 替换内容
        if self.use_csse:
            print("\n[CSSE - Channel-Spatial-Spectrum Expert]")
            print("  Encoder:")

            # ===== E2 Bias 配置（全层一致，便于做消融/论文叙事）=====
            # conf: 让 E2 的 logit 偏置随 alpha_a（A-gate）逐步生效（更符合“训练早期不依赖A，后期逐步引入A”的叙事）
            csse_bias_cfg = dict(
                e2_bias_mode="conf",        # "const" / "conf" / None
                e2_bias_k=0.6,              # 推荐起跑 0.4~0.8
                e2_gate_with_alpha=True,    # True: bias受alpha_a调制；False: 纯常量偏置
                # e2_logit_bias=0.0,        # 如果你在CSSE/CSSERouter也保留了这个参数，可显式写上
            )

    # Encoder
            self.csse_enc1 = CSSE(
                channels=128, num_proto=num_prototypes,
                enable_expert3=False,
                enable_expert4=False,
                **csse_bias_cfg
            )  # x2

            self.csse_enc2 = CSSE(
                channels=256, num_proto=num_prototypes,
                enable_expert3=False,
                enable_expert4=False,
                **csse_bias_cfg
            )  # x3

            self.csse_enc3 = CSSE(
                channels=512, num_proto=num_prototypes,
                enable_expert3=False,
                enable_expert4=False,
                **csse_bias_cfg
            )  # x4

            print("  Decoder:")

            # Decoder
            self.csse_dec1 = CSSE(
                channels=512, num_proto=num_prototypes,
                enable_expert3=False,
                enable_expert4=False,
                **csse_bias_cfg
            )  # up1 output

            self.csse_dec2 = CSSE(
                channels=256, num_proto=num_prototypes,
                enable_expert3=False,
                enable_expert4=False,
                **csse_bias_cfg
            )  # up2 output

            self.csse_dec3 = CSSE(
                channels=128, num_proto=num_prototypes,
                enable_expert3=False,
                enable_expert4=False,
                **csse_bias_cfg
            )  # up3 output

            
        
        # 4. PGAC++模块（跳跃连接）
        if self.use_pgacpp:
            print("\n[PGAC++ - Abundance-Guided Channel Gating]")
            self.pgac1 = PGACPP(in_channels=64, num_proto=num_prototypes)
            self.pgac2 = PGACPP(in_channels=128, num_proto=num_prototypes)
            self.pgac3 = PGACPP(in_channels=256, num_proto=num_prototypes)
            self.pgac4 = PGACPP(in_channels=512, num_proto=num_prototypes)
            self.pgac5 = PGACPP(in_channels=1024, num_proto=num_prototypes)
        
        if any([self.use_spgapp, self.use_csse, self.use_pgacpp]):
            print("="*70 + "\n")
        
        # ========== 解码器 ==========
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        self.out_conv = OutConv(64, out_channels)
    
    def forward(self, x, return_unmixing=False):
        """
        前向传播（完整PPEF++框架）
        
        Args:
            x: (B, 40, H, W) 高光谱输入
            return_unmixing: 是否返回unmixing输出（训练/验证时需要）
        
        Returns:
            如果return_unmixing=False:
                out: (B, 1, H, W) 分割输出
            如果return_unmixing=True:
                out: (B, 1, H, W) 分割输出
                A2: (B, K, H, W) 丰度图
                X2_hat: (B, num_bands, H, W) 重建光谱
                X2_down: (B, num_bands, H, W) 下采样的真实光谱
        """
        # 保存原始输入（用于unmixing head）
        x_hsi = x  # (B, 40, H, W)
        
        # ========== 编码器 ==========
        x1 = self.in_conv(x)    # (B, 64, H, W)
        x2 = self.down1(x1)     # (B, 128, H/2, W/2)
        x3 = self.down2(x2)     # (B, 256, H/4, W/4)
        
        # ========== 光谱解混头：生成丰度图A2 ==========
        A2, X2_hat, X2_down = self.unmix_head(x3, x_hsi)
        # A2: (B, K, H/4, W/4) 丰度图
        # X2_hat: (B, 40, H/4, W/4) 重建光谱
        # X2_down: (B, 40, H/4, W/4) 下采样的真实光谱
        
        # ========== 生成各尺度的丰度图（从A2插值）==========
        A0 = F.interpolate(A2, size=x1.shape[-2:], mode='bilinear', align_corners=False)  # (B,K,H,W)
        A1 = F.interpolate(A2, size=x2.shape[-2:], mode='bilinear', align_corners=False)  # (B,K,H/2,W/2)
        # A2 本身就是 (B,K,H/4,W/4)
        
        # ========== 编码器增强 ==========
        # x2增强
        if self.use_csse:
            x2 = self.csse_enc1(x2, A1)
        
        # x3增强（SPGA++ + CSSE）
        if self.use_spgapp:
            x3 = self.spga3(x3, A2)
        if self.use_csse:
            x3 = self.csse_enc2(x3, A2)
        
        # x4增强
        x4 = self.down3(x3)     # (B, 512, H/8, W/8)
        A3 = F.interpolate(A2, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        if self.use_spgapp:
            x4 = self.spga4(x4, A3)
        if self.use_csse:
            x4 = self.csse_enc3(x4, A3)
        
        # x5
        x5 = self.down4(x4)     # (B, 1024, H/16, W/16)
        A4 = F.interpolate(A2, size=x5.shape[-2:], mode='bilinear', align_corners=False)
        
        # ========== 跳跃连接（PGAC++门控）==========
        # 修改：浅层(s1,s2)不用PGAC++，只在深层(s3,s4,s5)使用
        # 避免A2在高分辨率特征上过度控制
        if self.use_pgacpp:
            s1 = x1  # 不使用PGAC++
            s2 = x2  # 不使用PGAC++
            s3 = self.pgac3(x3, A2)  # 只在深层使用
            s4 = self.pgac4(x4, A3)
            s5 = self.pgac5(x5, A4)
        else:
            s1, s2, s3, s4, s5 = x1, x2, x3, x4, x5
        
        # ========== 解码器（带CSSE增强）==========
        d4 = self.up1(x5, s4)   # (B, 512, H/8, W/8)
        if self.use_csse:
            d4 = self.csse_dec1(d4, A3)
        
        d3 = self.up2(d4, s3)   # (B, 256, H/4, W/4)
        if self.use_csse:
            d3 = self.csse_dec2(d3, A2)
        
        d2 = self.up3(d3, s2)   # (B, 128, H/2, W/2)
        if self.use_csse:
            d2 = self.csse_dec3(d2, A1)
        
        d1 = self.up4(d2, s1)   # (B, 64, H, W)
        
        # ========== 输出 ==========
        out = self.out_conv(d1)  # (B, 1, H, W)
        
        # 根据需要返回unmixing输出
        if return_unmixing:
            return out, A2, X2_hat, X2_down 
        else:
            return out
    
    def get_prototypes(self, to_cpu=False):
        """
        获取学到的光谱原型（用于可视化和分析）
        
        Args:
            to_cpu: 是否拷贝到CPU（默认False）
        
        Returns:
            prototypes: (K, num_bands) 光谱原型
        """
        return self.unmix_head.get_prototypes(to_cpu=to_cpu)


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing UNet_PPEF++ Model")
    print("="*70)
    
    # 测试参数
    B = 2
    in_channels = 40
    out_channels = 1
    H, W = 256, 256  # 假设输入尺寸
    num_prototypes = 4
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {B}")
    print(f"  Input channels (spectral bands): {in_channels}")
    print(f"  Output channels: {out_channels}")
    print(f"  Input size: ({H}, {W})")
    print(f"  Num prototypes: {num_prototypes}")
    
    # 创建模型
    print(f"\n{'='*70}")
    print("Initializing model...")
    print(f"{'='*70}")
    
    model = UNet_PPEFPP(
        in_channels=in_channels,
        out_channels=out_channels,
        num_prototypes=num_prototypes,
        use_spgapp=True,
        use_csse=True,
        use_pgacpp=True
    )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}")
    print(f"Model statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory (FP32): ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"{'='*70}")
    
    # 创建输入
    x = torch.randn(B, in_channels, H, W)
    print(f"\nInput shape: {x.shape}")
    
    # 前向传播
    print("\nForward pass...")
    model.eval()
    with torch.no_grad():
        out = model(x)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, out_channels, H, W), "Output shape mismatch!"
    
    # 获取解混输出
    A2, X2_hat, X2_down = model.get_unmixing_outputs()
    print(f"\nUnmixing outputs:")
    print(f"  A2 (abundance map): {A2.shape}")
    print(f"  X2_hat (reconstructed spectrum): {X2_hat.shape}")
    print(f"  X2_down (downsampled true spectrum): {X2_down.shape}")
    
    # 验证丰度图性质
    A2_sum = A2.sum(dim=1)
    print(f"\nAbundance map properties:")
    print(f"  Sum along prototype dim: mean={A2_sum.mean():.4f}, std={A2_sum.std():.4f}")
    
    # 获取原型
    prototypes = model.get_prototypes()
    print(f"\nPrototypes shape: {prototypes.shape}")
    print(f"Prototype norms: {prototypes.norm(dim=1)}")
    
    # 梯度测试
    print(f"\n{'='*70}")
    print("Gradient test...")
    print(f"{'='*70}")
    
    model.train()
    x_test = torch.randn(1, in_channels, H, W, requires_grad=True)
    out_test = model(x_test)
    loss = out_test.sum()
    loss.backward()
    
    print(f"✓ Backward pass successful")
    print(f"✓ Input gradient norm: {x_test.grad.norm().item():.6f}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)






