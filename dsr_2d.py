"""
DSR - Dynamic Spectral Routing Module (2D版本)
动态光谱路由模块

迁移自nnUNet v2的3D实现，适配到2D高光谱图像分割

核心创新：
1. 多专家路由系统 - 4个专家网络，各有专长（通道、空间、细粒度、标准）
2. 动态路由机制 - 基于特征自动选择处理路径
3. 自适应特征聚合 - 软路由机制，允许特征经过多条路径
4. 轻量级设计 - 使用深度可分离卷积大幅减少参数

关键转换（3D → 2D）：
- 输入: (B, C, H, W, D) → (B, C, H, W)
- 光谱维度D → 通道维度C（60通道包含光谱信息）
- Conv3d → Conv2d (深度可分离)
- 专家类型适配：光谱专家 → 通道专家
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRouter(nn.Module):
    """
    动态路由器 (2D适配版)
    
    创新点：基于输入特征动态生成路由权重，决定每个专家的贡献
    
    3D→2D转换：
    - 路由决策从光谱特征 → 全局通道统计
    - 使用全局平均池化提取特征
    """
    def __init__(self, channels, num_experts=4, temperature=1.0):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        
        # 可学习的温度参数（用于控制路由的稀疏性）
        self.temperature_param = nn.Parameter(torch.tensor(temperature))
        
        # 路由决策网络（轻量级）
        hidden_dim = max(channels // 4, num_experts * 2)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Flatten(),
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts, bias=False),
        )
        
    def forward(self, x):
        """
        输入: x (B, C, H, W)
        输出: routing_weights (B, num_experts)
        """
        # 生成路由权重
        routing_logits = self.router(x)  # (B, num_experts)
        
        # Softmax with temperature (温度越高，分布越平滑)
        routing_weights = F.softmax(routing_logits / self.temperature_param, dim=1)  # (B, num_experts)
        
        return routing_weights


class LightweightExpert(nn.Module):
    """
    轻量级专家网络 (2D适配版 - 深度可分离卷积)
    
    创新点：每个专家专注于处理特定类型的特征模式
    使用深度可分离卷积大幅减少参数量
    
    专家类型：
    - channel: 通道专家（类似3D的光谱专家）- 1x1卷积
    - spatial: 空间专家 - 3x3深度可分离卷积
    - fine_grained: 细粒度专家 - 3x3深度可分离卷积
    - standard: 标准专家 - 1x1卷积
    """
    def __init__(self, channels, expert_type='standard'):
        super().__init__()
        self.expert_type = expert_type
        self.channels = channels
        
        if expert_type == 'channel':
            # 通道专家：专注于通道间交互（类似3D的光谱专家）
            # 使用1x1卷积
            self.expert_net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            
        elif expert_type == 'spatial':
            # 空间专家：专注于空间维度
            # 使用深度可分离卷积（Depthwise + Pointwise）
            self.expert_net = nn.Sequential(
                # Depthwise: 每个通道独立进行空间卷积
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, 
                         groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                # Pointwise: 通道间融合
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            
        elif expert_type == 'fine_grained':
            # 细粒度专家：捕获细节特征
            # 使用深度可分离卷积
            self.expert_net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, 
                         groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            
        else:  # 'standard'
            # 标准专家：通用处理
            # 使用1x1卷积
            self.expert_net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x):
        """
        输入: x (B, C, H, W)
        输出: processed_x (B, C, H, W)
        """
        return self.expert_net(x)


class DSRModule2D(nn.Module):
    """
    完整的DSR模块 (2D适配版 - 完整版，4个专家)
    
    创新点总结：
    1. 多专家系统 - 4个专家网络，各有专长
    2. 动态路由 - 基于特征自动选择处理路径
    3. 软路由机制 - 允许特征经过多条路径（不是硬路由）
    4. 轻量级设计 - 深度可分离卷积减少参数
    
    输入: (B, C, H, W)
    输出: (B, C, H, W)  # 尺度保持不变
    
    应用位置（对应nnUNet的stage 1,2,3）：
    - down1后: 128通道 (512x640分辨率)
    - down2后: 256通道 (256x320分辨率)
    - down3后: 512通道 (128x160分辨率)
    """
    def __init__(self, 
                 channels, 
                 num_experts=4,
                 expert_types=None,
                 temperature=1.0,
                 use_residual=True,
                 dropout_rate=0.1):
        super().__init__()
        
        self.channels = channels
        self.num_experts = num_experts
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        
        # 默认的专家类型配置
        if expert_types is None:
            expert_types = ['channel', 'spatial', 'fine_grained', 'standard']
        assert len(expert_types) == num_experts, "expert_types数量必须等于num_experts"
        
        # 1. 动态路由器
        self.router = DynamicRouter(channels, num_experts, temperature)
        
        # 2. 多个专家网络
        self.experts = nn.ModuleList([
            LightweightExpert(channels, expert_type=expert_types[i])
            for i in range(num_experts)
        ])
        
        # 3. 输出增强（可选）
        self.output_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # 4. Dropout层（防止过拟合）
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        print(f"  [DSR-2D] Initialized:")
        print(f"    - Channels: {channels}")
        print(f"    - Num experts: {num_experts}")
        print(f"    - Expert types: {expert_types}")
        print(f"    - Temperature: {temperature}")
        print(f"    - Use residual: {use_residual}")
    
    def forward(self, x):
        """
        输入: x (B, C, H, W)
        输出: routed_x (B, C, H, W)
        
        处理流程：
        1. 路由器决定每个专家的权重
        2. 所有专家并行处理输入
        3. 根据路由权重聚合专家输出
        4. 残差连接
        """
        identity = x  # 残差连接
        B, C, H, W = x.shape
        
        # Step 1: 生成路由权重
        routing_weights = self.router(x)  # (B, num_experts)
        
        # Step 2: 所有专家并行处理
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (B, C, H, W)
            expert_outputs.append(expert_out)
        
        # Step 3: 加权聚合专家输出
        # routing_weights: (B, num_experts) -> (B, num_experts, 1, 1, 1)
        weights_expanded = routing_weights.view(B, self.num_experts, 1, 1, 1)
        
        # Stack experts: (num_experts, B, C, H, W) -> (B, num_experts, C, H, W)
        stacked_experts = torch.stack(expert_outputs, dim=1)  # (B, num_experts, C, H, W)
        
        # Weighted sum: (B, num_experts, C, H, W) * (B, num_experts, 1, 1, 1)
        weighted = stacked_experts * weights_expanded  # (B, num_experts, C, H, W)
        aggregated = weighted.sum(dim=1)  # (B, C, H, W)
        
        # Step 4: 输出增强
        output = self.output_enhance(aggregated)
        
        # Step 4.5: Dropout（防止过拟合）
        if self.dropout is not None and self.training:
            output = self.dropout(output)
        
        # Step 5: 残差连接
        if self.use_residual:
            output = output + identity
        
        return output
    
    def get_routing_weights(self, x):
        """
        获取路由权重（用于分析和可视化）
        
        返回: routing_weights (B, num_experts)
        """
        with torch.no_grad():
            routing_weights = self.router(x)
        return routing_weights.cpu()
    
    def get_routing_statistics(self, x):
        """
        获取路由统计信息（用于分析哪个专家被激活）
        
        返回: dict of {expert_name: avg_weight}
        """
        routing_weights = self.get_routing_weights(x)
        avg_weights = routing_weights.mean(dim=0).numpy()  # (num_experts,)
        
        expert_names = ['Channel', 'Spatial', 'Fine-grained', 'Standard']
        return dict(zip(expert_names, avg_weights))


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing DSR 2D Module (完整版，4个专家)")
    print("="*70)
    
    # 测试参数（模拟实际使用场景）
    test_configs = [
        (2, 128, 512, 640),   # down1后: 128通道
        (2, 256, 256, 320),   # down2后: 256通道
        (2, 512, 128, 160),   # down3后: 512通道
    ]
    
    for i, (B, C, H, W) in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"Test {i+1}: (B={B}, C={C}, H={H}, W={W})")
        print(f"{'='*70}")
        
        # 创建模块
        dsr = DSRModule2D(channels=C, num_experts=4)
        
        # 创建输入
        x = torch.randn(B, C, H, W)
        print(f"Input shape: {x.shape}")
        
        # 前向传播
        with torch.no_grad():
            out = dsr(x)
        
        print(f"Output shape: {out.shape}")
        assert out.shape == x.shape, "Shape mismatch!"
        
        # 获取路由统计
        routing_stats = dsr.get_routing_statistics(x)
        print(f"Routing statistics:")
        for expert_name, weight in routing_stats.items():
            print(f"  {expert_name:15s}: {weight:.4f}")
        print(f"  Sum: {sum(routing_stats.values()):.4f} (should be ~1.0)")
        
        print(f"✓ Test {i+1} passed!")
    
    # 参数量统计
    print(f"\n{'='*70}")
    print("Parameter Statistics (for 256 channels):")
    print(f"{'='*70}")
    dsr_256 = DSRModule2D(256, num_experts=4)
    total_params = sum(p.numel() for p in dsr_256.parameters())
    trainable_params = sum(p.numel() for p in dsr_256.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Memory (FP32): ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 梯度测试
    print(f"\n{'='*70}")
    print("Gradient Test:")
    print(f"{'='*70}")
    dsr_test = DSRModule2D(128, num_experts=4)
    dsr_test.train()
    x_test = torch.randn(2, 128, 256, 256, requires_grad=True)
    out_test = dsr_test(x_test)
    loss = out_test.sum()
    loss.backward()
    
    print(f"✓ Backward pass successful")
    print(f"✓ Input gradient shape: {x_test.grad.shape}")
    print(f"✓ Routing weights gradient exists: {dsr_test.router.router[-1].weight.grad is not None}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)

