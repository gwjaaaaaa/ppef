"""
EPFD - Expert-Prior Fusion Decoder (2D版本)
专家-先验融合解码器

迁移自nnUNet v2的3D实现，适配到2D高光谱图像分割

核心创新：
1. 将encoder的DSR专家知识迁移到decoder
2. 用Value Network（RL思想）评估每个专家的价值
3. 动态融合4个专家 + 光谱先验
4. 实现encoder→decoder的知识流动闭环

应用位置：
- 解码器前3层：在concat之后，conv之前进行专家融合

关键转换（3D → 2D）：
- 输入: decoder_feat (B,C,H,W,D), skip_feat (B,C,H,W,D) → (B,C,H,W)
- 输出: (B, 2C, H, W, D) → (B, 2C, H, W)
- Conv3d → Conv2d (深度可分离)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class LightweightExpert(nn.Module):
    """
    超轻量专家网络 (2D适配版)
    继承DSR的专家"风格"但参数极少
    """
    def __init__(self, channels: int, expert_type: str):
        super().__init__()
        self.expert_type = expert_type
        
        if expert_type == 'channel':
            # 通道专家：1x1卷积（通道间交互）
            self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False, groups=max(channels//4, 1))
        elif expert_type == 'spatial':
            # 空间专家：深度可分离卷积
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            )
        elif expert_type == 'fine':
            # 细粒度专家：3x3 深度可分离
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            )
        else:  # standard
            # 标准专家：1x1卷积
            self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * 0.1  # 小权重，避免主导融合


class ValueNetwork(nn.Module):
    """
    超轻量价值网络 (2D适配版 - RL思想)
    
    评估每个专家对当前解码任务的价值（期望贡献）
    
    灵感：
    - RL中的Value Function: V(s) = 期望累积回报
    - 这里：V(expert_i | state) = expert_i对分割质量的期望提升
    
    输入状态：
    - decoder当前特征
    - skip特征
    
    输出：
    - 每个专家的价值分数 ∈ R^4
    """
    def __init__(self, channels: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        
        # 极简状态编码器：直接pool + linear，无卷积
        hidden_dim = max(16, channels // 8)  # 进一步压缩
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels * 2, hidden_dim, bias=False)
        self.value_head = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"      ├─ Value Network params: {total_params:,} (~{total_params/1000:.1f}K)")
    
    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_feat: (B, C, H, W) decoder特征
            skip_feat: (B, C, H, W) skip特征
        Returns:
            values: (B, num_experts) 每个专家的价值分数
        """
        B = decoder_feat.shape[0]
        
        # Pool + flatten
        d_pooled = self.pool(decoder_feat).view(B, -1)  # (B, C)
        s_pooled = self.pool(skip_feat).view(B, -1)     # (B, C)
        state = torch.cat([d_pooled, s_pooled], dim=1)  # (B, 2C)
        
        # FC编码
        hidden = F.relu(self.fc(state), inplace=True)  # (B, hidden_dim)
        
        # 价值输出
        values = self.value_head(hidden)  # (B, num_experts)
        
        return values


class PriorGate(nn.Module):
    """
    超轻量光谱先验门控 (2D适配版)
    
    用光谱先验调制专家融合的结果
    """
    def __init__(self, channels: int, spectral_dim: int = 40):
        super().__init__()
        self.channels = channels
        
        # 极简先验投影（单层，无bias）
        self.prior_proj = nn.Linear(spectral_dim, channels, bias=False)
        
        # 简化门控：只用pool + fc
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate_fc = nn.Linear(channels, channels, bias=False)
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"      ├─ Prior Gate params: {total_params:,} (~{total_params/1000:.1f}K)")
    
    def forward(self, x: torch.Tensor, spectral_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 专家融合后的特征
            spectral_prior: (spectral_dim,) 光谱先验权重
        Returns:
            out: (B, C, H, W) 先验调制后的特征
        """
        B, C = x.shape[:2]
        
        if spectral_prior is not None:
            # 投影先验到通道空间
            w_prior = torch.sigmoid(self.prior_proj(spectral_prior.to(x.device)))  # (C,)
            w_prior = w_prior.view(1, C, 1, 1)  # (1, C, 1, 1)
        else:
            w_prior = 1.0
        
        # 自适应门控
        x_pooled = self.pool(x).view(B, C)  # (B, C)
        w_gate = torch.sigmoid(self.gate_fc(x_pooled)).view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # 融合
        out = x * w_prior * w_gate
        
        return out


class EPFDModule2D(nn.Module):
    """
    Expert-Prior Fusion Decoder (2D适配版)
    
    完整的专家-先验融合解码模块
    
    工作流程：
    1. 4个轻量专家分别处理decoder+skip的concat特征
    2. Value Network评估每个专家的价值
    3. 基于价值加权融合专家输出
    4. 用光谱先验调制融合结果
    
    输入: decoder_feat (B,C,H,W), skip_feat (B,C,H,W)
    输出: (B, 2C, H, W) 融合后的特征（for decoder.conv）
    """
    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        spectral_dim: int = 40,
        spectral_prior_path: Optional[str] = None,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            channels: 特征通道数（decoder+skip concat后的通道数，即2C）
            num_experts: 专家数量（默认4）
            spectral_dim: 光谱维度（默认60）
            spectral_prior_path: 光谱先验权重文件路径
            dropout_rate: Dropout概率（默认0.1）
        """
        super().__init__()
        
        self.channels = channels
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate
        
        # === 极简降维和升维 ===
        # 输入是2C（decoder+skip concat），降到C//4（极度压缩），专家处理，再升回2C
        self.input_channels = channels  # 实际输入是2C
        self.expert_channels = max(channels // 4, 16)  # 专家在C//4上工作，最小16
        
        # 使用普通1x1卷积（去掉group以避免shape不匹配）
        self.dim_reduction = nn.Conv2d(channels, self.expert_channels, 
                                       kernel_size=1, bias=False)
        self.dim_expansion = nn.Conv2d(self.expert_channels, channels, 
                                       kernel_size=1, bias=False)
        
        # === 4个轻量专家（在expert_channels上工作）===
        expert_types = ['channel', 'spatial', 'fine', 'standard']
        self.experts = nn.ModuleList([
            LightweightExpert(self.expert_channels, expert_type) 
            for expert_type in expert_types[:num_experts]
        ])
        
        # === Value Network（基于原始通道数C，不是expert_channels）===
        # decoder_feat和skip_feat都是C通道，所以Value Network输入是2C
        original_C = channels // 2  # 输入是2C，原始单边是C
        self.value_net = ValueNetwork(original_C, num_experts)
        
        # === 光谱先验门控（在专家通道上工作）===
        self.prior_gate = PriorGate(self.expert_channels, spectral_dim)
        
        # === 加载光谱先验 ===
        self.register_buffer('spectral_prior', torch.ones(spectral_dim))
        if spectral_prior_path is not None:
            self._load_spectral_prior(spectral_prior_path)
        
        # === Dropout层（防止过拟合）===
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        # 统计参数
        expert_params = sum(sum(p.numel() for p in expert.parameters()) 
                           for expert in self.experts)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"    [EPFD-2D] C={channels}, Experts={num_experts}")
        print(f"      ├─ Expert params: {expert_params:,} (~{expert_params/1000:.1f}K)")
        # Value Network和Prior Gate的参数已在各自__init__中打印
        print(f"      └─ Total params: {total_params:,} (~{total_params/1000:.1f}K)")
    
    def _load_spectral_prior(self, path: str):
        """加载光谱先验权重"""
        try:
            weights = np.load(path)
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            self.spectral_prior.copy_(torch.from_numpy(weights).float())
            print(f"      └─ Spectral prior loaded from {path}")
        except Exception as e:
            print(f"      └─ Warning: Failed to load spectral prior: {e}")
    
    def forward(
        self,
        decoder_feat: torch.Tensor,
        skip_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播（轻量化版本）
        
        Args:
            decoder_feat: (B, C, H, W) decoder上采样后的特征
            skip_feat: (B, C, H, W) skip连接特征（已经过PGAC）
        Returns:
            out: (B, 2C, H, W) 融合后的特征（for decoder.conv）
        """
        B, C, H, W = decoder_feat.shape
        
        # === Step 1: Concat decoder和skip ===
        x_concat = torch.cat([decoder_feat, skip_feat], dim=1)  # (B, 2C, H, W)
        
        # === Step 2: 降维（2C → expert_channels）===
        x_reduced = self.dim_reduction(x_concat)  # (B, expert_channels, H, W)
        
        # === Step 3: 4个专家分别处理 ===
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x_reduced)  # (B, expert_channels, H, W)
            expert_outputs.append(expert_out)
        
        # === Step 4: Value Network评估专家价值 ===
        values = self.value_net(decoder_feat, skip_feat)  # (B, num_experts)
        
        # Softmax归一化为权重
        weights = F.softmax(values, dim=1)  # (B, num_experts)
        
        # === Step 5: 加权融合专家输出 ===
        weights = weights.view(B, self.num_experts, 1, 1, 1)  # (B, num_experts, 1, 1, 1)
        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, num_experts, expert_channels, H, W)
        
        expert_fusion = (expert_stack * weights).sum(dim=1)  # (B, expert_channels, H, W)
        
        # === Step 6: 光谱先验调制 ===
        out = self.prior_gate(expert_fusion, self.spectral_prior)  # (B, expert_channels, H, W)
        
        # === Step 7: 升维（expert_channels → 2C）+ 残差连接 ===
        out_expanded = self.dim_expansion(out)  # (B, 2C, H, W)
        
        # === Step 7.5: Dropout（防止过拟合）===
        if self.dropout is not None and self.training:
            out_expanded = self.dropout(out_expanded)
        
        # 残差连接
        out_final = out_expanded + x_concat
        
        return out_final


def build_epfd_module(
    channels: int,
    num_experts: int = 4,
    spectral_dim: int = 40,
    spectral_prior_path: Optional[str] = None
) -> EPFDModule2D:
    """
    构建EPFD模块
    
    Args:
        channels: 通道数（concat后的通道数，即2C）
        num_experts: 专家数量
        spectral_dim: 光谱维度
        spectral_prior_path: 光谱先验路径
    Returns:
        EPFD模块实例
    """
    return EPFDModule2D(
        channels=channels,
        num_experts=num_experts,
        spectral_dim=spectral_dim,
        spectral_prior_path=spectral_prior_path
    )


# 测试代码
if __name__ == "__main__":
    print("="*80)
    print("Testing EPFD 2D Module")
    print("="*80)
    
    # 测试参数（模拟解码器的不同stage）
    test_configs = [
        (2, 512, 128, 160),    # up1: 1024通道concat后
        (2, 256, 256, 320),    # up2: 512通道concat后
        (2, 128, 512, 640),    # up3: 256通道concat后
    ]
    
    for i, (B, C, H, W) in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"Test {i+1}: (B={B}, C={C}, H={H}, W={W})")
        print(f"{'='*80}")
        
        # C是单边通道数，concat后是2C
        epfd = build_epfd_module(
            channels=C*2,  # concat后的通道数
            num_experts=4,
            spectral_dim=40
        )
        
        # 模拟decoder和skip特征
        decoder_feat = torch.randn(B, C, H, W)
        skip_feat = torch.randn(B, C, H, W)
        
        print(f"Decoder feat: {decoder_feat.shape}")
        print(f"Skip feat: {skip_feat.shape}")
        
        # 前向传播
        with torch.no_grad():
            out = epfd(decoder_feat, skip_feat)
        
        print(f"Output: {out.shape}")
        assert out.shape == (B, C*2, H, W), f"Shape mismatch! Expected ({B}, {C*2}, {H}, {W}), got {out.shape}"
        print(f"✓ Test {i+1} passed!")
    
    # 梯度测试
    print(f"\n{'='*80}")
    print("Gradient Test:")
    print(f"{'='*80}")
    epfd_test = build_epfd_module(256, 4)
    epfd_test.train()
    d_test = torch.randn(2, 128, 128, 160, requires_grad=True)
    s_test = torch.randn(2, 128, 128, 160, requires_grad=True)
    out_test = epfd_test(d_test, s_test)
    loss = out_test.sum()
    loss.backward()
    
    print(f"✓ Backward pass successful")
    print(f"✓ Decoder feat gradient: {d_test.grad is not None}")
    print(f"✓ Skip feat gradient: {s_test.grad is not None}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

