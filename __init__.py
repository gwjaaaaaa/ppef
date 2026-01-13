"""
PPEF (Prototype-Prior-Expert Fusion) Modules for 2D UNet
原型-先验-专家融合模块 (2D适配版)

迁移自 nnUNet v2 的3D实现，适配到2D高光谱图像分割

完整框架包含4个模块：
- SPGA: 光谱原型引导自适应注意力（编码器）
- DSR: 动态光谱路由（编码器）
- PGAC: 原型引导自适应通道注意力（跳跃连接）
- EPFD: 专家-先验融合解码器（解码器）

PPEF++ (Enhanced Version) 包含增强模块：
- SpectralUnmixingHead: 光谱解混头（生成丰度图）
- CSSE: 通道-空间-光谱专家模块（替代DSR）
- PGACPP: 基于丰度图的通道门控（替代PGAC）
- SPGAPP: 基于丰度图的增强注意力（替代SPGA）
"""

from .spga_2d import SPGAModule2D
from .dsr_2d import DSRModule2D
from .pgac_2d import PGACModule2D
from .epfd_2d import EPFDModule2D

# PPEF++ 新模块
from .unmixing_head import SpectralUnmixingHead
from .csse import CSSE
from .pgacpp import PGACPP
from .spgapp import SPGAPP

__all__ = [
    # 原始PPEF模块
    'SPGAModule2D',
    'DSRModule2D',
    'PGACModule2D',
    'EPFDModule2D',
    # PPEF++增强模块
    'SpectralUnmixingHead',
    'CSSE',
    'PGACPP',
    'SPGAPP',
]

__version__ = '2.0.0'

