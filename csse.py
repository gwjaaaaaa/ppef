# """
# CSSE - Channel-Spatial Spectrum Expert Module
# 通道-空间-光谱专家模块

# 核心创新：
# 1. 4个专家网络 - 各有专长（光谱局部、光谱全局、空间边缘、空间区域）
# 2. 联合路由器 - 基于特征F和丰度图A共同决策
# 3. 编码/解码共用 - 同一套专家在编码器和解码器都使用
# 4. 可解释性 - 可视化哪个专家被激活

# 专家类型：
# - Expert 1: Spectral-Local（光谱局部）- 组卷积捕获局部光谱模式
# - Expert 2: Spectral-Global（光谱全局）- SE机制+丰度引导
# - Expert 3: Spatial-Edge（空间边缘）- 深度可分离卷积+Laplacian高通滤波
# - Expert 4: Spatial-Region（空间区域）- 多尺度膨胀卷积

# 应用位置：
# - 编码器：x2, x3, x4 后各加一个CSSE
# - 解码器：up1, up2, up3 输出后各加一个CSSE
# """

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SpectralLocalExpert(nn.Module):
#     """
#     Expert 1: Spectral-Local（光谱局部专家）
    
#     专长：捕获局部通道间的相关性
#     方法：组卷积（Group Convolution）
#     """
#     def __init__(self, channels: int, groups: int = 4):
#         super().__init__()
#         C_mid = channels // 2
        
#         self.net = nn.Sequential(
#             # 1x1 降维
#             nn.Conv2d(channels, C_mid, kernel_size=1, bias=False),
#             nn.BatchNorm2d(C_mid),
#             nn.ReLU(inplace=True),
            
#             # 组卷积：将通道分组，每组内部做1x1卷积
#             nn.Conv2d(C_mid, C_mid, kernel_size=1, groups=groups, bias=False),
#             nn.BatchNorm2d(C_mid),
#             nn.ReLU(inplace=True),
            
#             # 1x1 升维
#             nn.Conv2d(C_mid, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#         )
    
#     def forward(self, F: torch.Tensor, A: torch.Tensor):
#         """F: (B,C,H,W), A: (B,K,H,W)"""
#         return self.net(F)


# class SpectralGlobalExpert(nn.Module):
#     """
#     [修改版] Expert 2: 像素级物理门控专家 (Pixel-wise Physical Gating Expert)
#     原名: SpectralGlobalExpert (保留类名以避免大量代码修改)
    
#     核心逻辑：
#     - 原来的 Expert 2 使用 GAP 压缩成 1 个点，与 Router 功能重复
#     - 新设计：不压缩图像，保留空间维度
#     - 将 K 通道的丰度图 A 通过卷积映射为 C 通道的"物理注意力图"
#     - 逐像素地校准特征 F（像素级门控）
    
#     机制: 
#     1. 不做 GAP，保留空间维度
#     2. 将丰度图 A (B, K, H, W) 映射为注意力图 (B, C, H, W)
#     3. 对特征图 F 进行逐像素门控 (Gating)
    
#     优势：
#     - Router: 全局规划（通过 GAP 看整体分布，决定专家权重）
#     - Expert 2: 局部执行（利用丰度图 A 在像素级精准控制）
#     - "Global Planning, Local Execution" - 不再冗余！
#     """
#     def __init__(self, channels: int, num_proto: int):
#         super().__init__()
        
#         # 将丰度图的 K 个通道映射到特征图的 C 个通道
#         # 使用 1x1 卷积实现通道对齐
#         self.phys_proj = nn.Sequential(
#             nn.Conv2d(num_proto, channels // 2, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False),
#             nn.Sigmoid()  # 生成 0~1 的门控权重
#         )
        
#         # 可选：对特征 F 也做一个简单的变换（轻量级）
#         self.feat_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels)
#         )

#         # 新加内容
#         self.beta = nn.Parameter(torch.tensor(0.0))
    
#     def forward(self, F: torch.Tensor, A: torch.Tensor):
#         """
#         F: (B, C, H, W) - 特征图
#         A: (B, K, H, W) - 丰度图（像素级物理信息）
        
#         Returns:
#             F_gated: (B, C, H, W) - 经过物理门控后的特征
#         """
#         # 1. 生成物理门控掩膜 (Physical Mask)
#         # mask shape: (B, C, H, W)
#         # 这里 A 是像素级对应的，保留了空间信息！
#         # phys_mask = self.phys_proj(A)
        
#         # # 2. 特征变换
#         # F_trans = self.feat_proj(F)
        
#         # # 3. 逐像素加权 (Hadamard Product)
#         # # 利用物理丰度信息，告诉特征图哪些像素该抑制，哪些该保留
#         # return F_trans * phys_mask
#         phys_mask = self.phys_proj(A)  # (B,C,H,W) in (0,1)
#         F_trans   = self.feat_proj(F)

#         # beta = F.softplus(self.beta)   # >=0，更稳定
#         # gain = 1.0 + beta * phys_mask
#         # return F_trans * gain

#         beta = torch.nn.functional.softplus(self.beta)  # ✅ 不会再被输入F覆盖
#         beta = beta.to(phys_mask.dtype)                 # ✅ AMP下更稳（可选但建议）
#         gain = 1.0 + beta * phys_mask
#         return F_trans * gain

#         # 等价写法：return F_trans + beta * (F_trans * phys_mask)


# class SpatialEdgeExpert(nn.Module):
#     """
#     Expert 3: Spatial-Edge（空间边缘专家）
    
#     专长：捕获空间边缘和高频细节
#     方法：深度可分离卷积 + Laplacian高通滤波
#     """
#     def __init__(self, channels: int):
#         super().__init__()
        
#         # 深度可分离卷积
#         self.dw_conv = nn.Conv2d(
#             channels, channels, kernel_size=3, 
#             padding=1, groups=channels, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(channels)
        
#         self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(channels)
        
#         # Laplacian算子（边缘检测）
#         lap_kernel = torch.tensor([
#             [0., -1., 0.],
#             [-1., 4., -1.],
#             [0., -1., 0.]
#         ]).view(1, 1, 3, 3)
#         self.register_buffer('lap_kernel', lap_kernel)
        
#         # 可学习的高通滤波强度
#         self.lap_alpha = nn.Parameter(torch.tensor(0.1))
    
#     def forward(self, F: torch.Tensor, A: torch.Tensor):
#         """F: (B,C,H,W), A: (B,K,H,W)"""
#         B, C, H, W = F.shape
        
#         # 深度可分离卷积
#         x = torch.relu(self.bn1(self.dw_conv(F)))
#         x = torch.relu(self.bn2(self.pw_conv(x)))
        
#         # Laplacian高通滤波（增强边缘）
#         # 对每个通道独立应用Laplacian
#         F_reshaped = F.view(B * C, 1, H, W)
#         lap_out = torch.nn.functional.conv2d(F_reshaped, self.lap_kernel, padding=1)
#         lap_out = lap_out.view(B, C, H, W)
        
#         # 结合原始特征和高通特征
#         return x + self.lap_alpha * lap_out


# class SpatialRegionExpert(nn.Module):
#     """
#     Expert 4: Spatial-Region（空间区域专家）
    
#     专长：捕获不同尺度的空间区域特征
#     方法：多尺度膨胀卷积
#     """
#     def __init__(self, channels: int):
#         super().__init__()
        
#         # 尺度1：dilation=1（小区域）
#         self.conv1 = nn.Conv2d(
#             channels, channels, kernel_size=3,
#             padding=1, dilation=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(channels)
        
#         # 尺度2：dilation=2（中等区域）
#         self.conv2 = nn.Conv2d(
#             channels, channels, kernel_size=3,
#             padding=2, dilation=2, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(channels)
        
#         # 融合
#         self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(channels)
    
#     def forward(self, F: torch.Tensor, A: torch.Tensor):
#         """F: (B,C,H,W), A: (B,K,H,W)"""
#         # 多尺度特征
#         y1 = torch.relu(self.bn1(self.conv1(F)))
#         y2 = torch.relu(self.bn2(self.conv2(F)))
        
#         # 融合
#         y = y1 + y2
#         y = torch.relu(self.bn3(self.conv3(y)))
        
#         return y


# class CSSERouter(nn.Module):
#     """
#     CSSE路由器：基于特征F和丰度图A联合决策
    
#     核心改进（Version 2.0）：
#     - F为主，A为辅：训练初期alpha_a≈0，路由只看F
#     - 可学习缩放：alpha_a随训练自适应，A逐渐参与
    
#     输入：
#     - F: (B, C, H, W) 特征图
#     - A: (B, K, H, W) 丰度图
    
#     输出：
#     - routing_weights: (B, 4) 4个专家的权重
#     """
#     def __init__(self, channels: int, num_proto: int, num_experts: int = 4, 
#         e2_index: int = None,                 # E2 在 active experts 里的索引（由CSSE自动传入）
#         e2_bias_mode: str = "none",           # "none" | "const" | "conf"
#         e2_logit_bias: float = 0.0,           # const 模式使用
#         e2_bias_k: float = 0.0,               # conf 模式使用：bias = k * conf
#         e2_gate_with_alpha: bool = True,      # 是否用 sigmoid(alpha_a) 做渐进门控（推荐 True）
#         eps: float = 1e-12):


#         super().__init__()
        
#         self.channels = channels
#         self.num_proto = num_proto
        
#         # 路由决策网络
#         hidden = max(channels // 2, 64)
#         self.fc1 = nn.Linear(channels + num_proto, hidden)
#         self.fc2 = nn.Linear(hidden, num_experts)
        
#         # A_g的可学习缩放系数，初始为0（训练初期不看A）
#         self.alpha_a = nn.Parameter(torch.tensor(0.0))

#         # 新增内容
#         self.e2_index = e2_index
#         self.e2_bias_mode = e2_bias_mode
#         self.e2_logit_bias = float(e2_logit_bias)
#         self.e2_bias_k = float(e2_bias_k)
#         self.e2_gate_with_alpha = bool(e2_gate_with_alpha)
#         self.eps = float(eps)
        
#         # 【新增】用于保存最后一次推理的权重（可视化用）
#         self.last_weights = None
    
#     def forward(self, F: torch.Tensor, A: torch.Tensor):
#         """
#         F: (B, C, H, W)
#         A: (B, K, H, W)
#         """
#         # 全局平均池化
#         f_g = F.mean(dim=(2, 3))  # (B, C)
#         a_g = A.mean(dim=(2, 3))  # (B, K)
        
#         # A_g加权：训练初期基本为0，随训练自适应
#         a_g_scaled = self.alpha_a * a_g
        
#         # 拼接
#         z = torch.cat([f_g, a_g_scaled], dim=1)  # (B, C+K)
        
#         # MLP生成路由logits
#         h = torch.relu(self.fc1(z))
#         logits = self.fc2(h)  # (B, num_experts)
        
#         # =========================================================
#         # ✅ 新增：对 E2 的 logit 加 bias（打破 collapse / 固化偏置）
#         # =========================================================
#         if (self.e2_index is not None) and (self.e2_bias_mode != "none"):
#             # gate：训练早期 alpha_a≈0 时，bias 也≈0；后期逐渐放开（更稳）
#             if self.e2_gate_with_alpha:
#                 gate = torch.sigmoid(self.alpha_a)  # 标量
#             else:
#                 gate = logits.new_tensor(1.0)

#             if self.e2_bias_mode == "const":
#                 # 常数偏置：bias = const * gate
#                 bias = logits.new_full((logits.size(0),), self.e2_logit_bias) * gate  # (B,)

#             elif self.e2_bias_mode == "conf":
#                 # 丰度置信度偏置：conf = 1 - H(a_g)/logK
#                 if self.num_proto <= 1:
#                     bias = logits.new_zeros((logits.size(0),))
#                 else:
#                     a_prob = a_g.clamp(min=self.eps)  # (B,K)
#                     ent = -(a_prob * a_prob.log()).sum(dim=1) / math.log(self.num_proto)  # (B,), in [0,1]
#                     conf = (1.0 - ent).clamp(0.0, 1.0)  # (B,)
#                     bias = (self.e2_bias_k * conf) * gate  # (B,)

#             else:
#                 raise ValueError(f"Unknown e2_bias_mode={self.e2_bias_mode}")

#             logits[:, self.e2_index] = logits[:, self.e2_index] + bias

#         # Softmax归一化
#         weights = torch.softmax(logits, dim=1)
        
#         # 【新增】保存最后一次的权重用于可视化（断开梯度，移到CPU）
#         # 只在评估模式下保存，避免训练时的开销
#         if not self.training:
#             self.last_weights = weights.detach().cpu().numpy()
        
#         return weights


# class AbundanceGuidedSpatialRouter(nn.Module):
#     """
#     方案A+: 丰度引导的空间自适应路由器
    
#     核心创新：
#     1. 从丰度图提取空间模式（边界、混合区域等）- 轻量设计
#     2. 端成员-专家亲和矩阵 - 学习物理先验
#     3. 通道感知 - 识别光谱特征显著性
#     4. 空间自适应 - 每个像素独立的专家权重
    
#     输入：
#     - F: (B, C, H, W) 特征图
#     - A: (B, K, H, W) 丰度图
    
#     输出：
#     - routing_weights: (B, num_experts, H, W) 空间自适应的专家权重
    
#     开销：
#     - 参数: ~12.5K (+0.1% vs 整个网络)
#     - FLOPs: ~30M per CSSE (+0.14% vs 整个网络)
#     """
#     def __init__(self, channels: int, num_proto: int, num_experts: int = 4):
#         super().__init__()
        
#         self.channels = channels
#         self.num_proto = num_proto
#         self.num_experts = num_experts
        
#         # ===== 创新1: 丰度空间模式提取 =====
#         # 从丰度图中提取边界、混合区域等空间信息
#         # 使用Depthwise conv保持轻量
#         self.abundance_spatial = nn.Sequential(
#             # DW: 每个端成员独立卷积（提取空间梯度）
#             nn.Conv2d(num_proto, num_proto, 3, padding=1, groups=num_proto, bias=False),
#             nn.BatchNorm2d(num_proto),
#             nn.ReLU(inplace=True),
#             # PW: 整合端成员信息
#             nn.Conv2d(num_proto, 16, 1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True)
#         )
#         # 参数: 4*9 + 4*16 = 100
        
#         # ===== 创新2: 通道感知（轻量级SE变体）=====
#         # 识别哪些通道（光谱波段）对路由决策重要
#         self.channel_aware = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, max(channels // 4, 32), 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(max(channels // 4, 32), max(channels // 4, 32), 1, bias=False),
#             nn.Sigmoid()
#         )
#         # 参数: C*C/4 + C/4*C/4 ≈ 5K
        
#         # ===== 创新3: 端成员-专家亲和矩阵 =====
#         # 学习"端成员组合 → 专家偏好"的物理先验
#         self.endmember_expert_affinity = nn.Parameter(
#             torch.randn(num_proto, num_experts) * 0.01
#         )
#         # 参数: K*E = 16
        
#         # ===== 主路由网络 =====
#         # 结合特征和丰度空间信息
#         self.main_router = nn.Sequential(
#             nn.Conv2d(channels + 16, 48, 1, bias=False),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(48, num_experts, 1, bias=False)
#         )
#         # 参数: (C+16)*48 + 48*E ≈ 7K
        
#         # 用于可视化和分析
#         self.last_weights = None
#         self.last_abundance_spatial = None
#         self.last_expert_prior = None
        
#         print(f"    [AbundanceGuidedSpatialRouter] Initialized:")
#         print(f"      - Params: ~{self._count_params()/1000:.1f}K")
#         print(f"      - Spatial routing: ✓ (每个像素独立权重)")
#         print(f"      - Abundance guidance: ✓ (空间模式提取)")
#         print(f"      - Affinity learning: ✓ ({num_proto}×{num_experts}矩阵)")
    
#     def _count_params(self):
#         """统计参数量"""
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
#     def forward(self, F: torch.Tensor, A: torch.Tensor, return_analysis=False):
#         """
#         F: (B, C, H, W) 特征图
#         A: (B, K, H, W) 丰度图
        
#         return_analysis: 是否返回中间结果用于可视化
#         """
#         B, C, H, W = F.shape
        
#         # ===== Step 1: 提取丰度的空间模式 =====
#         # 这里提取边界、混合区域等信息
#         A_spatial = self.abundance_spatial(A)  # (B, 16, H, W)
        
#         # ===== Step 2: 通道感知（可选，可用于后续分析）=====
#         channel_weight = self.channel_aware(F)  # (B, C/4, 1, 1)
        
#         # ===== Step 3: 丰度先验（端成员 → 专家偏好）=====
#         # 基于全局丰度分布生成专家偏好
#         A_global = A.mean(dim=(2, 3))  # (B, K)
#         expert_prior = A_global @ self.endmember_expert_affinity  # (B, E)
#         expert_prior = expert_prior.view(B, self.num_experts, 1, 1)  # broadcast
        
#         # ===== Step 4: 主路由（空间自适应）=====
#         # 结合特征和丰度空间信息
#         FA = torch.cat([F, A_spatial], dim=1)  # (B, C+16, H, W)
#         logits = self.main_router(FA)  # (B, E, H, W)
        
#         # ===== Step 5: 融合先验 =====
#         # 加入丰度先验（可调整权重）
#         logits = logits + 0.3 * expert_prior
        
#         # ===== Step 6: Softmax归一化 =====
#         weights = torch.softmax(logits, dim=1)  # (B, E, H, W)
        
#         # ===== 保存用于可视化 =====
#         if not self.training:
#             self.last_weights = weights.detach()
#             self.last_abundance_spatial = A_spatial.detach()
#             self.last_expert_prior = expert_prior.detach()
        
#         if return_analysis:
#             analysis = {
#                 'weights': weights.detach(),
#                 'abundance_spatial': A_spatial.detach(),
#                 'expert_prior': expert_prior.detach(),
#                 'affinity_matrix': self.endmember_expert_affinity.detach(),
#                 'channel_importance': channel_weight.detach()
#             }
#             return weights, analysis
        
#         return weights


# class CSSE(nn.Module):
#     """
#     CSSE - Channel-Spatial Spectrum Expert Module
    
#     完整的CSSE模块，包含4个专家和一个路由器
    
#     输入：
#     - F: (B, C, H, W) 特征图
#     - A: (B, K, H, W) 丰度图
    
#     输出：
#     - enhanced_F: (B, C, H, W) 增强后的特征
    
#     工作流程：
#     1. 路由器基于F和A生成4个专家的权重
#     2. 4个专家并行处理输入F（都可以访问A）
#     3. 加权聚合专家输出
#     4. 残差连接
    
#     消融实验支持：
#     - enable_expert1/2/3/4: 控制每个专家的开关（True=启用，False=禁用）
#     """
#     def __init__(self, channels: int, num_proto: int, groups: int = 4,
#                  enable_expert1: bool = True,
#                  enable_expert2: bool = True,
#                  enable_expert3: bool = True,
#                  enable_expert4: bool = False,
#                  # ===== 新增：E2 bias 控制 =====
#                  e2_bias_mode: str = "none",       # "none" | "const" | "conf"
#                  e2_logit_bias: float = 0.0,       # const 用
#                  e2_bias_k: float = 0.0,           # conf 用
#                  e2_gate_with_alpha: bool = True   # 推荐 True
#                  ):
#         super().__init__()
        
#         self.channels = channels
#         self.num_proto = num_proto
#         self.groups = groups
        
#         # 专家开关
#         self.enable_expert1 = enable_expert1
#         self.enable_expert2 = enable_expert2
#         self.enable_expert3 = enable_expert3
#         self.enable_expert4 = enable_expert4
        
#         # 统计启用的专家数量
#         self.num_active_experts = sum([enable_expert1, enable_expert2, enable_expert3, enable_expert4])
        
#         # 创建专家列表（只包含启用的专家）
#         self.experts = nn.ModuleList()
#         self.expert_names = []  # 用于可视化
        
#         if enable_expert1:
#             self.experts.append(SpectralLocalExpert(channels, groups))
#             self.expert_names.append('expert1_spectral_local')
#         if enable_expert2:
#             self.experts.append(SpectralGlobalExpert(channels, num_proto))
#             self.expert_names.append('expert2_spectral_global')
#         if enable_expert3:
#             self.experts.append(SpatialEdgeExpert(channels))
#             self.expert_names.append('expert3_spatial_edge')
#         if enable_expert4:
#             self.experts.append(SpatialRegionExpert(channels))
#             self.expert_names.append('expert4_spatial_region')
        
#         # 新增内容
#         # E2 在 active experts 中的位置（若E2未启用则为None）
#         e2_index = None
#         if 'expert2_spectral_global' in self.expert_names:
#             e2_index = self.expert_names.index('expert2_spectral_global')

        
#         # # 路由器（只在有多个专家时需要）
#         # if self.num_active_experts > 1:
#         #     self.router = CSSERouter(channels, num_proto, num_experts=self.num_active_experts)
#         # else:
#         #     self.router = None

#         # 替换上述路由器创建
#         if self.num_active_experts > 1:
#             self.router = CSSERouter(
#                 channels, num_proto,
#                 num_experts=self.num_active_experts,
#                 e2_index=e2_index,
#                 e2_bias_mode=e2_bias_mode,
#                 e2_logit_bias=e2_logit_bias,
#                 e2_bias_k=e2_bias_k,
#                 e2_gate_with_alpha=e2_gate_with_alpha
#             )
#         else:
#             self.router = None

        
#         # 输出缩放参数（可学习）- 从0.1开始，让专家做轻微修正
#         self.gamma = nn.Parameter(torch.tensor(0.1))
        
#         # 打印专家配置
#         expert_status = f"E1={'✓' if enable_expert1 else '✗'} E2={'✓' if enable_expert2 else '✗'} E3={'✓' if enable_expert3 else '✗'} E4={'✓' if enable_expert4 else '✗'}"
#         print(f"    [CSSE] C={channels}, K={num_proto}, Groups={groups}, {expert_status}, Active={self.num_active_experts}")
    
#     def forward(self, F: torch.Tensor, A: torch.Tensor):
#         """
#         F: (B, C, H, W) 特征图
#         A: (B, K, H, W) 丰度图
#         """
#         B, C, H, W = F.shape
        
#         # 如果没有启用任何专家，直接返回原始特征
#         if self.num_active_experts == 0:
#             return F
        
#         # ========== Step 1: 专家并行处理 ==========
#         # 使用ModuleList中的专家（只包含启用的专家）
#         expert_outputs = []
#         for expert in self.experts:
#             expert_outputs.append(expert(F, A))  # (B, C, H, W)
        
#         # ========== Step 2: 路由决策与加权聚合 ==========
#         if self.num_active_experts == 1:
#             # 只有一个专家，直接使用其输出
#             F_agg = expert_outputs[0]
#         else:
#             # 多个专家时需要路由
#             w = self.router(F, A)  # (B, num_active_experts) - 现在是3个专家
            
#             # 堆叠专家输出
#             E = torch.stack(expert_outputs, dim=1)  # (B, num_active_experts, C, H, W)
            
#             # 加权聚合
#             w = w.view(B, self.num_active_experts, 1, 1, 1)  # (B, num_active_experts, 1, 1, 1)
#             F_agg = (E * w).sum(dim=1)  # (B, C, H, W)
        
#         # ========== Step 3: 残差连接 ==========
#         return F + self.gamma * F_agg
    
#     def get_routing_weights(self, F: torch.Tensor, A: torch.Tensor):
#         """
#         获取路由权重（用于可视化和分析）
        
#         Returns:
#             (B, num_active_experts) tensor，权重和严格为1
#         """
#         with torch.no_grad():
#             if self.router is not None:
#                 weights = self.router(F, A)  # (B, num_active_experts)
#             else:
#                 # 只有一个专家时，返回权重 [1.0]
#                 weights = torch.ones(F.shape[0], self.num_active_experts, device=F.device)
#         return weights.cpu()
    
#     def get_expert_weights(self):
#         """
#         获取路由器最后一次的专家权重（用于可视化）
        
#         Returns:
#             (B, num_active_experts) numpy array，如果没有路由器则返回None
#         """
#         if self.router is not None and hasattr(self.router, 'last_weights'):
#             return self.router.last_weights
#         return None


# # 测试代码
# if __name__ == "__main__":
#     print("="*70)
#     print("Testing CSSE Module")
#     print("="*70)
    
#     # 测试参数（编码器和解码器的不同通道数）
#     test_configs = [
#         # 编码器
#         (2, 128, 128, 128, 4, "Encoder: down1 -> x2"),
#         (2, 256, 64, 64, 4, "Encoder: down2 -> x3"),
#         (2, 512, 32, 32, 4, "Encoder: down3 -> x4"),
#         # 解码器
#         (2, 512, 32, 32, 4, "Decoder: up1 output"),
#         (2, 256, 64, 64, 4, "Decoder: up2 output"),
#         (2, 128, 128, 128, 4, "Decoder: up3 output"),
#     ]
    
#     for i, (B, C, H, W, K, desc) in enumerate(test_configs):
#         print(f"\n{'='*70}")
#         print(f"Test {i+1}: {desc}")
#         print(f"  Shape: (B={B}, C={C}, H={H}, W={W}), K={K}")
#         print(f"{'='*70}")
        
#         # 创建模块
#         csse = CSSE(channels=C, num_proto=K, groups=4)
        
#         # 创建输入
#         F = torch.randn(B, C, H, W)
#         A = torch.randn(B, K, H, W)
#         A = F.softmax(A, dim=1)  # 确保A是归一化的丰度图
        
#         print(f"Input shapes:")
#         print(f"  F: {F.shape}")
#         print(f"  A: {A.shape}")
        
#         # 前向传播
#         with torch.no_grad():
#             out = csse(F, A)
        
#         print(f"Output shape: {out.shape}")
#         assert out.shape == F.shape, "Shape mismatch!"
        
#         # 获取路由权重
#         routing_weights = csse.get_routing_weights(F, A)
#         print(f"Routing weights (avg over batch):")
#         avg_weights = routing_weights.mean(dim=0).numpy()
#         expert_names = ['Spectral-Local', 'Spectral-Global', 'Spatial-Edge', 'Spatial-Region']
#         for name, weight in zip(expert_names, avg_weights):
#             print(f"  {name:20s}: {weight:.4f}")
#         print(f"  Sum: {avg_weights.sum():.4f} (should be 1.0)")
        
#         print(f"✓ Test {i+1} passed!")
    
#     # 参数量统计
#     print(f"\n{'='*70}")
#     print("Parameter Statistics:")
#     print(f"{'='*70}")
    
#     for C in [128, 256, 512]:
#         csse = CSSE(C, num_proto=4, groups=4)
#         total_params = sum(p.numel() for p in csse.parameters())
#         trainable_params = sum(p.numel() for p in csse.parameters() if p.requires_grad)
#         print(f"  C={C:3d}: {total_params:>8,} params (~{total_params/1000:.1f}K)")
    
#     # 梯度测试
#     print(f"\n{'='*70}")
#     print("Gradient Test:")
#     print(f"{'='*70}")
#     csse_test = CSSE(128, num_proto=4)
#     csse_test.train()
    
#     F_test = torch.randn(2, 128, 64, 64, requires_grad=True)
#     A_test = F.softmax(torch.randn(2, 4, 64, 64), dim=1)
    
#     out_test = csse_test(F_test, A_test)
#     loss = out_test.sum()
#     loss.backward()
    
#     print(f"✓ Backward pass successful")
#     print(f"✓ Input gradient norm: {F_test.grad.norm().item():.6f}")
#     print(f"✓ Gamma grad: {csse_test.gamma.grad.item():.6f}")
    
#     print("\n" + "="*70)
#     print("✅ All tests passed!")
#     print("="*70)
"""
CSSE - Channel-Spatial Spectrum Expert Module
通道-空间-光谱专家模块

核心创新：
1. 4个专家网络 - 各有专长（光谱局部、光谱全局、空间边缘、空间区域）
2. 联合路由器 - 基于特征F和丰度图A共同决策
3. 编码/解码共用 - 同一套专家在编码器和解码器都使用
4. 可解释性 - 可视化哪个专家被激活

专家类型：
- Expert 1: Spectral-Local（光谱局部）- 组卷积捕获局部光谱模式
- Expert 2: Spectral-Global（光谱全局）- SE机制+丰度引导
- Expert 3: Spatial-Edge（空间边缘）- 深度可分离卷积+Laplacian高通滤波
- Expert 4: Spatial-Region（空间区域）- 多尺度膨胀卷积

应用位置：
- 编码器：x2, x3, x4 后各加一个CSSE
- 解码器：up1, up2, up3 输出后各加一个CSSE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLocalExpert(nn.Module):
    """
    Expert 1: Spectral-Local（光谱局部专家）
    
    专长：捕获局部通道间的相关性
    方法：组卷积（Group Convolution）
    """
    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        C_mid = channels // 2
        
        self.net = nn.Sequential(
            # 1x1 降维
            nn.Conv2d(channels, C_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_mid),
            nn.ReLU(inplace=True),
            
            # 组卷积：将通道分组，每组内部做1x1卷积
            nn.Conv2d(C_mid, C_mid, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(C_mid),
            nn.ReLU(inplace=True),
            
            # 1x1 升维
            nn.Conv2d(C_mid, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, F: torch.Tensor, A: torch.Tensor):
        """F: (B,C,H,W), A: (B,K,H,W)"""
        return self.net(F)


class SpectralGlobalExpert(nn.Module):
    """
    [修改版] Expert 2: 像素级物理门控专家 (Pixel-wise Physical Gating Expert)
    原名: SpectralGlobalExpert (保留类名以避免大量代码修改)
    
    核心逻辑：
    - 原来的 Expert 2 使用 GAP 压缩成 1 个点，与 Router 功能重复
    - 新设计：不压缩图像，保留空间维度
    - 将 K 通道的丰度图 A 通过卷积映射为 C 通道的"物理注意力图"
    - 逐像素地校准特征 F（像素级门控）
    
    机制: 
    1. 不做 GAP，保留空间维度
    2. 将丰度图 A (B, K, H, W) 映射为注意力图 (B, C, H, W)
    3. 对特征图 F 进行逐像素门控 (Gating)
    
    优势：
    - Router: 全局规划（通过 GAP 看整体分布，决定专家权重）
    - Expert 2: 局部执行（利用丰度图 A 在像素级精准控制）
    - "Global Planning, Local Execution" - 不再冗余！
    """
    def __init__(self, channels: int, num_proto: int):
        super().__init__()
        
        # 将丰度图的 K 个通道映射到特征图的 C 个通道
        # 使用 1x1 卷积实现通道对齐
        self.phys_proj = nn.Sequential(
            nn.Conv2d(num_proto, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # 生成 0~1 的门控权重
        )
        
        # 可选：对特征 F 也做一个简单的变换（轻量级）
        self.feat_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 新加内容
        self.beta = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, F: torch.Tensor, A: torch.Tensor):
        """
        F: (B, C, H, W) - 特征图
        A: (B, K, H, W) - 丰度图（像素级物理信息）
        
        Returns:
            F_gated: (B, C, H, W) - 经过物理门控后的特征
        """
        # 1. 生成物理门控掩膜 (Physical Mask)
        # mask shape: (B, C, H, W)
        # 这里 A 是像素级对应的，保留了空间信息！
        # phys_mask = self.phys_proj(A)
        
        # # 2. 特征变换
        # F_trans = self.feat_proj(F)
        
        # # 3. 逐像素加权 (Hadamard Product)
        # # 利用物理丰度信息，告诉特征图哪些像素该抑制，哪些该保留
        # return F_trans * phys_mask
        phys_mask = self.phys_proj(A)  # (B,C,H,W) in (0,1)
        F_trans   = self.feat_proj(F)

        # beta = F.softplus(self.beta)   # >=0，更稳定
        # gain = 1.0 + beta * phys_mask
        # return F_trans * gain

        beta = torch.nn.functional.softplus(self.beta)  # ✅ 不会再被输入F覆盖
        beta = beta.to(phys_mask.dtype)                 # ✅ AMP下更稳（可选但建议）
        gain = 1.0 + beta * phys_mask
        return F_trans * gain

        # 等价写法：return F_trans + beta * (F_trans * phys_mask)


class SpatialEdgeExpert(nn.Module):
    """
    Expert 3: Spatial-Edge（空间边缘专家）
    
    专长：捕获空间边缘和高频细节
    方法：深度可分离卷积 + Laplacian高通滤波
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # 深度可分离卷积
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3, 
            padding=1, groups=channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Laplacian算子（边缘检测）
        lap_kernel = torch.tensor([
            [0., -1., 0.],
            [-1., 4., -1.],
            [0., -1., 0.]
        ]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)
        
        # 可学习的高通滤波强度
        self.lap_alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, F: torch.Tensor, A: torch.Tensor):
        """F: (B,C,H,W), A: (B,K,H,W)"""
        B, C, H, W = F.shape
        
        # 深度可分离卷积
        x = torch.relu(self.bn1(self.dw_conv(F)))
        x = torch.relu(self.bn2(self.pw_conv(x)))
        
        # Laplacian高通滤波（增强边缘）
        # 对每个通道独立应用Laplacian
        F_reshaped = F.view(B * C, 1, H, W)
        lap_out = torch.nn.functional.conv2d(F_reshaped, self.lap_kernel, padding=1)
        lap_out = lap_out.view(B, C, H, W)
        
        # 结合原始特征和高通特征
        return x + self.lap_alpha * lap_out


class SpatialRegionExpert(nn.Module):
    """
    Expert 4: Spatial-Region（空间区域专家）
    
    专长：捕获不同尺度的空间区域特征
    方法：多尺度膨胀卷积
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # 尺度1：dilation=1（小区域）
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=1, dilation=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 尺度2：dilation=2（中等区域）
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=2, dilation=2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        
        # 融合
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
    
    def forward(self, F: torch.Tensor, A: torch.Tensor):
        """F: (B,C,H,W), A: (B,K,H,W)"""
        # 多尺度特征
        y1 = torch.relu(self.bn1(self.conv1(F)))
        y2 = torch.relu(self.bn2(self.conv2(F)))
        
        # 融合
        y = y1 + y2
        y = torch.relu(self.bn3(self.conv3(y)))
        
        return y


class CSSERouter(nn.Module):
    """
    CSSE路由器：基于特征F和丰度图A联合决策
    
    核心改进（Version 2.0）：
    - F为主，A为辅：训练初期alpha_a≈0，路由只看F
    - 可学习缩放：alpha_a随训练自适应，A逐渐参与
    
    输入：
    - F: (B, C, H, W) 特征图
    - A: (B, K, H, W) 丰度图
    
    输出：
    - routing_weights: (B, 4) 4个专家的权重
    """
    def __init__(self, channels: int, num_proto: int, num_experts: int = 4, 
        e2_index: int = None,                 # E2 在 active experts 里的索引（由CSSE自动传入）
        e2_bias_mode: str = "none",           # "none" | "const" | "conf"
        e2_logit_bias: float = 0.0,           # const 模式使用
        e2_bias_k: float = 0.0,               # conf 模式使用：bias = k * conf
        e2_gate_with_alpha: bool = True,      # 是否用 sigmoid(alpha_a) 做渐进门控（推荐 True）
        eps: float = 1e-12):


        super().__init__()
        
        self.channels = channels
        self.num_proto = num_proto
        
        # 路由决策网络
        hidden = max(channels // 2, 64)
        self.fc1 = nn.Linear(channels + num_proto, hidden)
        self.fc2 = nn.Linear(hidden, num_experts)
        
        # A_g的可学习缩放系数，初始为0（训练初期不看A）
        self.alpha_a = nn.Parameter(torch.tensor(0.0))

        # 新增内容
        self.e2_index = e2_index
        self.e2_bias_mode = e2_bias_mode
        self.e2_logit_bias = float(e2_logit_bias)
        self.e2_bias_k = float(e2_bias_k)
        self.e2_gate_with_alpha = bool(e2_gate_with_alpha)
        self.eps = float(eps)
        
        # 【新增】用于保存最后一次推理的权重（可视化用）
        self.last_weights = None
    
    def forward(self, F: torch.Tensor, A: torch.Tensor):
        """
        F: (B, C, H, W)
        A: (B, K, H, W)
        """
        # 全局平均池化
        f_g = F.mean(dim=(2, 3))  # (B, C)
        a_g = A.mean(dim=(2, 3))  # (B, K)
        
        # A_g加权：训练初期基本为0，随训练自适应
        a_g_scaled = self.alpha_a * a_g
        
        # 拼接
        z = torch.cat([f_g, a_g_scaled], dim=1)  # (B, C+K)
        
        # MLP生成路由logits
        h = torch.relu(self.fc1(z))
        logits = self.fc2(h)  # (B, num_experts)
        
        # =========================================================
        # ✅ 新增：对 E2 的 logit 加 bias（打破 collapse / 固化偏置）
        # =========================================================
        if (self.e2_index is not None) and (self.e2_bias_mode != "none"):
            # gate：训练早期 alpha_a≈0 时，bias 也≈0；后期逐渐放开（更稳）
            if self.e2_gate_with_alpha:
                gate = torch.sigmoid(self.alpha_a)  # 标量
            else:
                gate = logits.new_tensor(1.0)

            if self.e2_bias_mode == "const":
                # 常数偏置：bias = const * gate
                bias = logits.new_full((logits.size(0),), self.e2_logit_bias) * gate  # (B,)

            elif self.e2_bias_mode == "conf":
                # 丰度置信度偏置：conf = 1 - H(a_g)/logK
                if self.num_proto <= 1:
                    bias = logits.new_zeros((logits.size(0),))
                else:
                    a_prob = a_g.clamp(min=self.eps)  # (B,K)
                    ent = -(a_prob * a_prob.log()).sum(dim=1) / math.log(self.num_proto)  # (B,), in [0,1]
                    conf = (1.0 - ent).clamp(0.0, 1.0)  # (B,)
                    bias = (self.e2_bias_k * conf) * gate  # (B,)

            else:
                raise ValueError(f"Unknown e2_bias_mode={self.e2_bias_mode}")

            logits[:, self.e2_index] = logits[:, self.e2_index] + bias

        # Softmax归一化
        weights = torch.softmax(logits, dim=1)
        
        # 【新增】保存最后一次的权重用于可视化（断开梯度，移到CPU）
        # 只在评估模式下保存，避免训练时的开销
        if not self.training:
            self.last_weights = weights.detach().cpu().numpy()
        
        return weights


class AbundanceGuidedSpatialRouter(nn.Module):
    """
    方案A+: 丰度引导的空间自适应路由器
    
    核心创新：
    1. 从丰度图提取空间模式（边界、混合区域等）- 轻量设计
    2. 端成员-专家亲和矩阵 - 学习物理先验
    3. 通道感知 - 识别光谱特征显著性
    4. 空间自适应 - 每个像素独立的专家权重
    
    输入：
    - F: (B, C, H, W) 特征图
    - A: (B, K, H, W) 丰度图
    
    输出：
    - routing_weights: (B, num_experts, H, W) 空间自适应的专家权重
    
    开销：
    - 参数: ~12.5K (+0.1% vs 整个网络)
    - FLOPs: ~30M per CSSE (+0.14% vs 整个网络)
    """
    def __init__(self, channels: int, num_proto: int, num_experts: int = 4):
        super().__init__()
        
        self.channels = channels
        self.num_proto = num_proto
        self.num_experts = num_experts
        
        # ===== 创新1: 丰度空间模式提取 =====
        # 从丰度图中提取边界、混合区域等空间信息
        # 使用Depthwise conv保持轻量
        self.abundance_spatial = nn.Sequential(
            # DW: 每个端成员独立卷积（提取空间梯度）
            nn.Conv2d(num_proto, num_proto, 3, padding=1, groups=num_proto, bias=False),
            nn.BatchNorm2d(num_proto),
            nn.ReLU(inplace=True),
            # PW: 整合端成员信息
            nn.Conv2d(num_proto, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # 参数: 4*9 + 4*16 = 100
        
        # ===== 创新2: 通道感知（轻量级SE变体）=====
        # 识别哪些通道（光谱波段）对路由决策重要
        self.channel_aware = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 32), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 32), max(channels // 4, 32), 1, bias=False),
            nn.Sigmoid()
        )
        # 参数: C*C/4 + C/4*C/4 ≈ 5K
        
        # ===== 创新3: 端成员-专家亲和矩阵 =====
        # 学习"端成员组合 → 专家偏好"的物理先验
        self.endmember_expert_affinity = nn.Parameter(
            torch.randn(num_proto, num_experts) * 0.01
        )
        # 参数: K*E = 16
        
        # ===== 主路由网络 =====
        # 结合特征和丰度空间信息
        self.main_router = nn.Sequential(
            nn.Conv2d(channels + 16, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_experts, 1, bias=False)
        )
        # 参数: (C+16)*48 + 48*E ≈ 7K
        
        # 用于可视化和分析
        self.last_weights = None
        self.last_abundance_spatial = None
        self.last_expert_prior = None
        
        print(f"    [AbundanceGuidedSpatialRouter] Initialized:")
        print(f"      - Params: ~{self._count_params()/1000:.1f}K")
        print(f"      - Spatial routing: ✓ (每个像素独立权重)")
        print(f"      - Abundance guidance: ✓ (空间模式提取)")
        print(f"      - Affinity learning: ✓ ({num_proto}×{num_experts}矩阵)")
    
    def _count_params(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, F: torch.Tensor, A: torch.Tensor, return_analysis=False):
        """
        F: (B, C, H, W) 特征图
        A: (B, K, H, W) 丰度图
        
        return_analysis: 是否返回中间结果用于可视化
        """
        B, C, H, W = F.shape
        
        # ===== Step 1: 提取丰度的空间模式 =====
        # 这里提取边界、混合区域等信息
        A_spatial = self.abundance_spatial(A)  # (B, 16, H, W)
        
        # ===== Step 2: 通道感知（可选，可用于后续分析）=====
        channel_weight = self.channel_aware(F)  # (B, C/4, 1, 1)
        
        # ===== Step 3: 丰度先验（端成员 → 专家偏好）=====
        # 基于全局丰度分布生成专家偏好
        A_global = A.mean(dim=(2, 3))  # (B, K)
        expert_prior = A_global @ self.endmember_expert_affinity  # (B, E)
        expert_prior = expert_prior.view(B, self.num_experts, 1, 1)  # broadcast
        
        # ===== Step 4: 主路由（空间自适应）=====
        # 结合特征和丰度空间信息
        FA = torch.cat([F, A_spatial], dim=1)  # (B, C+16, H, W)
        logits = self.main_router(FA)  # (B, E, H, W)
        
        # ===== Step 5: 融合先验 =====
        # 加入丰度先验（可调整权重）
        logits = logits + 0.3 * expert_prior
        
        # ===== Step 6: Softmax归一化 =====
        weights = torch.softmax(logits, dim=1)  # (B, E, H, W)
        
        # ===== 保存用于可视化 =====
        if not self.training:
            self.last_weights = weights.detach()
            self.last_abundance_spatial = A_spatial.detach()
            self.last_expert_prior = expert_prior.detach()
        
        if return_analysis:
            analysis = {
                'weights': weights.detach(),
                'abundance_spatial': A_spatial.detach(),
                'expert_prior': expert_prior.detach(),
                'affinity_matrix': self.endmember_expert_affinity.detach(),
                'channel_importance': channel_weight.detach()
            }
            return weights, analysis
        
        return weights


class CSSE(nn.Module):
    """
    CSSE - Channel-Spatial Spectrum Expert Module
    
    完整的CSSE模块，包含4个专家和一个路由器
    
    输入：
    - F: (B, C, H, W) 特征图
    - A: (B, K, H, W) 丰度图
    
    输出：
    - enhanced_F: (B, C, H, W) 增强后的特征
    
    工作流程：
    1. 路由器基于F和A生成4个专家的权重
    2. 4个专家并行处理输入F（都可以访问A）
    3. 加权聚合专家输出
    4. 残差连接
    
    消融实验支持：
    - enable_expert1/2/3/4: 控制每个专家的开关（True=启用，False=禁用）
    """
    def __init__(self, channels: int, num_proto: int, groups: int = 4,
                 enable_expert1: bool = True,
                 enable_expert2: bool = True,
                 enable_expert3: bool = True,
                 enable_expert4: bool = False,
                 # ===== 新增：E2 bias 控制 =====
                 e2_bias_mode: str = "none",       # "none" | "const" | "conf"
                 e2_logit_bias: float = 0.0,       # const 用
                 e2_bias_k: float = 0.0,           # conf 用
                 e2_gate_with_alpha: bool = True,  # 推荐 True
                 force_uniform_routing: bool = False  # ✅ 新增：强制均匀路由
                 ):
        super().__init__()
        
        self.channels = channels
        self.num_proto = num_proto
        self.groups = groups
        
        # 专家开关
        self.enable_expert1 = enable_expert1
        self.enable_expert2 = enable_expert2
        self.enable_expert3 = enable_expert3
        self.enable_expert4 = enable_expert4
        
        # 统计启用的专家数量
        self.num_active_experts = sum([enable_expert1, enable_expert2, enable_expert3, enable_expert4])

        self.force_uniform_routing = force_uniform_routing  # ✅
        self.last_weights = None  # ✅ 新增：CSSE级别缓存（不依赖router）
        
        # 创建专家列表（只包含启用的专家）
        self.experts = nn.ModuleList()
        self.expert_names = []  # 用于可视化
        
        if enable_expert1:
            self.experts.append(SpectralLocalExpert(channels, groups))
            self.expert_names.append('expert1_spectral_local')
        if enable_expert2:
            self.experts.append(SpectralGlobalExpert(channels, num_proto))
            self.expert_names.append('expert2_spectral_global')
        if enable_expert3:
            self.experts.append(SpatialEdgeExpert(channels))
            self.expert_names.append('expert3_spatial_edge')
        if enable_expert4:
            self.experts.append(SpatialRegionExpert(channels))
            self.expert_names.append('expert4_spatial_region')
        
        # 新增内容
        # E2 在 active experts 中的位置（若E2未启用则为None）
        e2_index = None
        if 'expert2_spectral_global' in self.expert_names:
            e2_index = self.expert_names.index('expert2_spectral_global')

        
        # # 路由器（只在有多个专家时需要）
        # if self.num_active_experts > 1:
        #     self.router = CSSERouter(channels, num_proto, num_experts=self.num_active_experts)
        # else:
        #     self.router = None

        # 替换上述路由器创建
        if self.num_active_experts > 1:
            self.router = CSSERouter(
                channels, num_proto,
                num_experts=self.num_active_experts,
                e2_index=e2_index,
                e2_bias_mode=e2_bias_mode,
                e2_logit_bias=e2_logit_bias,
                e2_bias_k=e2_bias_k,
                e2_gate_with_alpha=e2_gate_with_alpha
            )
        else:
            self.router = None

        
        # 输出缩放参数（可学习）- 从0.1开始，让专家做轻微修正
        self.gamma = nn.Parameter(torch.tensor(0.1))
        
        # 打印专家配置
        expert_status = f"E1={'✓' if enable_expert1 else '✗'} E2={'✓' if enable_expert2 else '✗'} E3={'✓' if enable_expert3 else '✗'} E4={'✓' if enable_expert4 else '✗'}"
        print(f"    [CSSE] C={channels}, K={num_proto}, Groups={groups}, {expert_status}, Active={self.num_active_experts}")
        
        # ===== for visualization =====
        self.debug_vis = False
        self.debug_cache = {}
    
    def forward(self, F: torch.Tensor, A: torch.Tensor):
        """
        F: (B, C, H, W) 特征图
        A: (B, K, H, W) 丰度图
        """
        B, C, H, W = F.shape
        
        # 如果没有启用任何专家，直接返回原始特征
        if self.num_active_experts == 0:
            return F
        
        # ========== Step 1: 专家并行处理 ==========
        # 使用ModuleList中的专家（只包含启用的专家）
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(F, A))  # (B, C, H, W)
        
        # ========== Step 2: 路由决策与加权聚合 ==========
        if self.num_active_experts == 1:
            # 只有一个专家，直接使用其输出
            F_agg = expert_outputs[0]
        else:
            # 多个专家时需要路由
            if self.force_uniform_routing:
                # ✅ 强制均匀：每个专家权重=1/E
                w = F.new_full((B, self.num_active_experts), 1.0 / self.num_active_experts)
            else:
                w = self.router(F, A)  # (B, num_active_experts)

            # ✅ 保存权重（用于可视化），避免强制均匀时 router.last_weights 为空
            if not self.training:
                self.last_weights = w.detach().cpu().numpy()

            
            # 堆叠专家输出
            E = torch.stack(expert_outputs, dim=1)  # (B, num_active_experts, C, H, W)
            
            # 加权聚合
            w = w.view(B, self.num_active_experts, 1, 1, 1)  # (B, num_active_experts, 1, 1, 1)
            F_agg = (E * w).sum(dim=1)  # (B, C, H, W)
            
            # ===== Debug: cache weighted contribution maps (adaptive weights) =====
            if (not self.training) and self.debug_vis:
                # E: (B,E,C,H,W), w: (B,E,1,1,1)
                contrib = (self.gamma * w) * E  # (B,E,C,H,W)

                # HxW map: abs + mean over channels (更稳)
                contrib_map = contrib.abs().mean(dim=2)  # (B,E,H,W)

                # 也可以顺便存一下未加权专家响应（可选）
                resp_map = E.abs().mean(dim=2)           # (B,E,H,W)

                self.debug_cache = {
                    "w": w.squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu(),    # (B,E)
                    "contrib_map": contrib_map.detach().cpu(),                    # (B,E,H,W)
                    "resp_map": resp_map.detach().cpu(),                          # (B,E,H,W)
                }
        
        # ========== Step 3: 残差连接 ==========
        return F + self.gamma * F_agg
    
    def get_routing_weights(self, F: torch.Tensor, A: torch.Tensor):
        """
        获取路由权重（用于可视化和分析）
        
        Returns:
            (B, num_active_experts) tensor，权重和严格为1
        """
        with torch.no_grad():
            if self.router is not None:
                if self.force_uniform_routing:
                    weights = F.new_full((F.shape[0], self.num_active_experts), 1.0 / self.num_active_experts)
                else:
                    weights = self.router(F, A)
            else:
                weights = torch.ones(F.shape[0], self.num_active_experts, device=F.device)
        return weights.cpu()
    
    def get_expert_weights(self):
        """
        获取路由器最后一次的专家权重（用于可视化）
        
        Returns:
            (B, num_active_experts) numpy array，如果没有路由器则返回None
        """
        # ✅ 优先返回 CSSE 自己保存的（强制均匀时也会有）
        if self.last_weights is not None:
            return self.last_weights

        # 其次再用 router 的
        if self.router is not None and hasattr(self.router, 'last_weights'):
            return self.router.last_weights
        return None
    
    def set_debug_vis(self, flag: bool = True):
        """设置可视化调试标志"""
        self.debug_vis = bool(flag)

    def get_debug_cache(self):
        """获取可视化调试缓存"""
        return self.debug_cache



# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing CSSE Module")
    print("="*70)
    
    # 测试参数（编码器和解码器的不同通道数）
    test_configs = [
        # 编码器
        (2, 128, 128, 128, 4, "Encoder: down1 -> x2"),
        (2, 256, 64, 64, 4, "Encoder: down2 -> x3"),
        (2, 512, 32, 32, 4, "Encoder: down3 -> x4"),
        # 解码器
        (2, 512, 32, 32, 4, "Decoder: up1 output"),
        (2, 256, 64, 64, 4, "Decoder: up2 output"),
        (2, 128, 128, 128, 4, "Decoder: up3 output"),
    ]
    
    for i, (B, C, H, W, K, desc) in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"Test {i+1}: {desc}")
        print(f"  Shape: (B={B}, C={C}, H={H}, W={W}), K={K}")
        print(f"{'='*70}")
        
        # 创建模块
        csse = CSSE(channels=C, num_proto=K, groups=4)
        
        # 创建输入
        F = torch.randn(B, C, H, W)
        A = torch.randn(B, K, H, W)
        A = F.softmax(A, dim=1)  # 确保A是归一化的丰度图
        
        print(f"Input shapes:")
        print(f"  F: {F.shape}")
        print(f"  A: {A.shape}")
        
        # 前向传播
        with torch.no_grad():
            out = csse(F, A)
        
        print(f"Output shape: {out.shape}")
        assert out.shape == F.shape, "Shape mismatch!"
        
        # 获取路由权重
        routing_weights = csse.get_routing_weights(F, A)
        print(f"Routing weights (avg over batch):")
        avg_weights = routing_weights.mean(dim=0).numpy()
        expert_names = ['Spectral-Local', 'Spectral-Global', 'Spatial-Edge', 'Spatial-Region']
        for name, weight in zip(expert_names, avg_weights):
            print(f"  {name:20s}: {weight:.4f}")
        print(f"  Sum: {avg_weights.sum():.4f} (should be 1.0)")
        
        print(f"✓ Test {i+1} passed!")
    
    # 参数量统计
    print(f"\n{'='*70}")
    print("Parameter Statistics:")
    print(f"{'='*70}")
    
    for C in [128, 256, 512]:
        csse = CSSE(C, num_proto=4, groups=4)
        total_params = sum(p.numel() for p in csse.parameters())
        trainable_params = sum(p.numel() for p in csse.parameters() if p.requires_grad)
        print(f"  C={C:3d}: {total_params:>8,} params (~{total_params/1000:.1f}K)")
    
    # 梯度测试
    print(f"\n{'='*70}")
    print("Gradient Test:")
    print(f"{'='*70}")
    csse_test = CSSE(128, num_proto=4)
    csse_test.train()
    
    F_test = torch.randn(2, 128, 64, 64, requires_grad=True)
    A_test = F.softmax(torch.randn(2, 4, 64, 64), dim=1)
    
    out_test = csse_test(F_test, A_test)
    loss = out_test.sum()
    loss.backward()
    
    print(f"✓ Backward pass successful")
    print(f"✓ Input gradient norm: {F_test.grad.norm().item():.6f}")
    print(f"✓ Gamma grad: {csse_test.gamma.grad.item():.6f}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
