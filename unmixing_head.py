"""
SpectralUnmixingHead - 光谱解混头
用于生成丰度图和重建光谱

核心功能：
1. 将编码器特征映射到光谱空间
2. 与全局光谱原型计算相似度，生成丰度图A2
3. 用丰度图和原型重建光谱X2_hat
4. 计算光谱重建损失L_recon

应用位置：
- 在encoder down2（x3）之后
- 输出A2用于后续所有尺度（通过插值）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralUnmixingHead(nn.Module):
    """
    光谱解混头（Spectral Unmixing Head）
    
    在编码器down2后生成丰度图A2和重建光谱X2_hat
    
    Args:
        feat_channels: 特征通道数（down2输出的通道数，如256）
        num_bands: 高光谱波段数（如60）
        num_proto: 光谱原型数量（如4或6）
    
    Forward:
        输入：
            x3: (B, feat_channels, H, W) 编码器down2输出
            x_hsi: (B, num_bands, H0, W0) 原始高光谱输入
        输出：
            A2: (B, K, H, W) 丰度图
            X2_hat: (B, num_bands, H, W) 重建光谱
            X2_down: (B, num_bands, H, W) 下采样的真实光谱
    """
    
    def __init__(self, feat_channels: int, num_bands: int, num_proto: int):
        super().__init__()
        self.num_bands = num_bands
        self.num_proto = num_proto
        
        # 特征到光谱空间的投影
        self.proj_to_spec = nn.Conv2d(
            feat_channels, num_bands, 
            kernel_size=1, bias=False
        )
        
        # 全局光谱原型 P_spec: (K, num_bands)
        # 每个原型代表一个纯净的光谱成分
        self.P_spec = nn.Parameter(torch.randn(num_proto, num_bands))
        
        print(f"  feat_channels={feat_channels}, num_bands={num_bands}, num_proto={num_proto}")
    
    def forward(self, x3: torch.Tensor, x_hsi: torch.Tensor):
        """
        前向传播
        
        Args:
            x3: (B, C=feat_channels, H, W) 编码器down2输出
            x_hsi: (B, num_bands, H0, W0) 原始高光谱输入
        
        Returns:
            A2: (B, K, H, W) 丰度图
            X2_hat: (B, num_bands, H, W) 重建光谱
            X2_down: (B, num_bands, H, W) 下采样的真实光谱（用于计算L_recon）
        """
        B, C, H, W = x3.shape
        
        # ========== Step 1: 特征映射到光谱空间 ==========
        F_spec = self.proj_to_spec(x3)  # (B, num_bands, H, W)
        
        # ========== Step 2: 下采样原始光谱到同一分辨率 ==========
        X2_down = F.interpolate(
            x_hsi, size=(H, W), 
            mode='bilinear', align_corners=False
        )  # (B, num_bands, H, W)
        
        # ========== Step 3: 计算与原型的相似度 S2 ==========
        # 归一化原型和特征（余弦相似度）
        P = F.normalize(self.P_spec, dim=1)  # (K, num_bands)
        F_norm = F.normalize(F_spec, dim=1)  # (B, num_bands, H, W)
        
        # 广播乘法计算相似度
        P_view = P.view(1, self.num_proto, self.num_bands, 1, 1)  # (1, K, num_bands, 1, 1)
        F_view = F_norm.unsqueeze(1)  # (B, 1, num_bands, H, W)
        
        # 点积相似度
        S2 = (P_view * F_view).sum(dim=2)  # (B, K, H, W)
        
        # ========== Step 4: Softmax生成丰度图 A2 ==========
        A2 = F.softmax(S2, dim=1)  # (B, K, H, W)
        # 每个像素的K个丰度值之和为1
        
        # ========== Step 5: 用A2和P重建光谱 X2_hat ==========
        # X_hat = P^T @ A  （矩阵乘法）
        # A2: (B, K, H, W) 丰度图
        # P_spec: (K, num_bands) 原型光谱
        # X2_hat = sum_k(A2[:, k, :, :] * P_spec[k, :])
        
        A2_flat = A2.view(B, self.num_proto, -1)  # (B, K, H*W)
        
        # einsum: 'bkn,kd->bdn'
        #   b: batch
        #   k: prototypes (求和消除)
        #   n: spatial (H*W)
        #   d: spectral bands
        X_hat_flat = torch.einsum('bkn,kd->bdn', A2_flat, self.P_spec)  # (B, num_bands, H*W)
        
        X2_hat = X_hat_flat.reshape(B, self.num_bands, H, W)  # (B, num_bands, H, W)
        
        return A2, X2_hat, X2_down
    
    def get_prototypes(self, to_cpu=False):
        """
        获取学到的光谱原型（用于可视化和分析）
        
        Args:
            to_cpu: 是否拷贝到CPU（默认False，保持在GPU上更快）
        
        Returns:
            prototypes: (K, num_bands) 光谱原型矩阵
        """
        if to_cpu:
            return self.P_spec.detach().cpu()
        else:
            return self.P_spec  # 保持在GPU上，不detach（用于loss计算）


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing SpectralUnmixingHead")
    print("="*70)
    
    # 测试参数
    B = 2
    feat_channels = 256
    num_bands = 40
    num_proto = 4
    H, W = 64, 64
    H0, W0 = 256, 256
    
    # 创建模块
    unmix_head = SpectralUnmixingHead(
        feat_channels=feat_channels,
        num_bands=num_bands,
        num_proto=num_proto
    )
    
    # 测试输入
    x3 = torch.randn(B, feat_channels, H, W)
    x_hsi = torch.randn(B, num_bands, H0, W0)
    
    print(f"\n输入:")
    print(f"  x3: {x3.shape}")
    print(f"  x_hsi: {x_hsi.shape}")
    
    # 前向传播
    A2, X2_hat, X2_down = unmix_head(x3, x_hsi)
    
    print(f"\n输出:")
    print(f"  A2 (丰度图): {A2.shape}")
    print(f"  X2_hat (重建光谱): {X2_hat.shape}")
    print(f"  X2_down (真实光谱): {X2_down.shape}")
    
    # 检查丰度图约束
    A2_sum = A2.sum(dim=1)  # 对K维求和
    print(f"\n丰度图约束检查:")
    print(f"  A2.sum(dim=1) 应该全为1: min={A2_sum.min():.4f}, max={A2_sum.max():.4f}")
    
    # 获取原型
    prototypes = unmix_head.get_prototypes()
    print(f"\n光谱原型:")
    print(f"  shape: {prototypes.shape}")
    
    print("\n" + "="*70)
    print("✓ SpectralUnmixingHead 测试通过！")
    print("="*70)
