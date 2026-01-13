import torch
from torch.utils.data import Dataset
import scipy.io
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


#DGA
class HyperspectralDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
       
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir is not None else image_dir
        self.transform = transform

        # 获取所有图像文件（不包含labels的.mat文件）
        self.hyperspectral_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.mat') and not f.endswith('labels.mat')])
        
        # 构建图像和标签的匹配对
        self.pairs = []
        for img_file in self.hyperspectral_files:

            # 标签文件名规则：尝试两种命名方式
            # 方式1: 同名（DGA512）: xxx.mat -> xxx.mat
            # 方式2: labels后缀（DGA_hpy94）: xxx.mat -> xxxlabels.mat

            base_name = img_file.replace('.mat', '')
            
            # 优先尝试labels.mat后缀
            mask_file_with_labels = f"{base_name}labels.mat"
            mask_file_same_name = img_file
            
            # 检查哪种命名方式存在
            if os.path.exists(os.path.join(self.mask_dir, mask_file_with_labels)):
                self.pairs.append((img_file, mask_file_with_labels))
            elif os.path.exists(os.path.join(self.mask_dir, mask_file_same_name)):
                self.pairs.append((img_file, mask_file_same_name))
        
        print(f'找到 {len(self.pairs)} 对高光谱图像和标签')
        assert len(self.pairs) > 0, f"未找到匹配的图像和掩码文件\n图像目录: {image_dir}\n标签目录: {self.mask_dir}"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # 获取文件名
        hyperspectral_filename, mask_filename = self.pairs[idx]

        # 加载高光谱图像
        hyperspectral_data = scipy.io.loadmat(os.path.join(self.image_dir, hyperspectral_filename))
        hyperspectral_image = hyperspectral_data['mat']

        # 加载掩码图像
        mask_data = scipy.io.loadmat(os.path.join(self.mask_dir, mask_filename))
        mask_image = mask_data['mat']
        
        # 将mask限制在[0, 1]范围内（二分类）
        mask_image = np.clip(mask_image, 0, 1)

        # 归一化：min-max归一化到[0, 1]
        hyperspectral_image = hyperspectral_image.astype(np.float32)
        # 全局归一化（基于整个图像的min/max）
        img_min = hyperspectral_image.min()
        img_max = hyperspectral_image.max()
        if img_max > img_min:
            hyperspectral_image = (hyperspectral_image - img_min) / (img_max - img_min)
        else:
            hyperspectral_image = hyperspectral_image * 0.0  # 全0图像

        # 转为 tensor
        hyperspectral_image = torch.tensor(hyperspectral_image, dtype=torch.float32).permute(2, 0, 1)
        mask_image = torch.tensor(mask_image.astype(np.float32), dtype=torch.float32)

        # 应用变换（如果有）
        if self.transform:
            hyperspectral_image = self.transform(hyperspectral_image)
            mask_image = self.transform(mask_image)

        # 获取图像基础名（无扩展名）
        image_name = os.path.splitext(hyperspectral_filename)[0]

        return hyperspectral_image, mask_image, image_name

#PLGC
class PLGCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): 高光谱图像所在的文件夹路径
            mask_dir (str): 掩码图像所在的文件夹路径
            transform (callable, optional): 用于对图像进行处理的转换操作
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 获取文件名（假设命名对应，比如 a12.mat 和 a12labels.mat）
        self.hyperspectral_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.mat')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.mat')])

        # 只保留文件名前缀匹配的组合
        self.pairs = []
        for img_file in self.hyperspectral_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}.mat"
            if mask_file in self.mask_files:
                self.pairs.append((img_file, mask_file))

        assert len(self.pairs) > 0, "未找到匹配的图像和掩码文件"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hyperspectral_filename, mask_filename = self.pairs[idx]

        # 加载高光谱图像
        hyperspectral_data = scipy.io.loadmat(os.path.join(self.image_dir, hyperspectral_filename))
        hyperspectral_image = hyperspectral_data['mat']#PLGA data;DGA512 mat

        # 加载掩码图像
        mask_data = scipy.io.loadmat(os.path.join(self.mask_dir, mask_filename))
        mask_image = mask_data['mat']

        mask_image = np.clip(mask_image, 0, 1)
        # 归一化：min-max归一化到[0, 1]
        hyperspectral_image = hyperspectral_image.astype(np.float32)
        img_min = hyperspectral_image.min()
        img_max = hyperspectral_image.max()
        if img_max > img_min:
            hyperspectral_image = (hyperspectral_image - img_min) / (img_max - img_min)
        else:
            hyperspectral_image = hyperspectral_image * 0.0

        # 转为 tensor，permute 调整为 (C, H, W)
        hyperspectral_image = torch.tensor(hyperspectral_image, dtype=torch.float32).permute(2, 0, 1)
        mask_image = torch.tensor(mask_image.astype(np.float32), dtype=torch.float32)

        # 应用变换（如果有）
        if self.transform:
            hyperspectral_image = self.transform(hyperspectral_image)
            mask_image = self.transform(mask_image)

        # 获取图像基础名（无扩展名和标签）
        image_name = os.path.splitext(hyperspectral_filename)[0]

        return hyperspectral_image, mask_image, image_name


# 支持指定keys的数据集（用于5折交叉验证）
class HyperspectralDatasetWithKeys(Dataset):
    def __init__(self, image_dir, mask_dir, keys, transform=None):
        """
        Args:
            image_dir (str): 高光谱图像所在的文件夹路径
            mask_dir (str): 掩码图像所在的文件夹路径
            keys (list): 要使用的图像文件名列表（不包含.mat后缀）
            transform (callable, optional): 用于对图像进行处理的转换操作
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.keys = keys
        
        # 构建图像和标签的匹配对（仅包含指定的keys）
        self.pairs = []
        for key in keys:
            img_file = f"{key}.mat"
            
            # 尝试两种标签文件命名方式
            # 方式1: labels后缀（DGA_hpy94）: xxx.mat -> xxxlabels.mat
            # 方式2: 同名（DGA512）: xxx.mat -> xxx.mat
            mask_file_with_labels = f"{key}labels.mat"
            mask_file_same_name = f"{key}.mat"
            
            # 检查图像文件是否存在
            img_path = os.path.join(image_dir, img_file)
            if not os.path.exists(img_path):
                print(f"警告：找不到图像文件 {img_file}")
                continue
            
            # 检查标签文件（优先使用labels后缀）
            mask_path_with_labels = os.path.join(mask_dir, mask_file_with_labels)
            mask_path_same_name = os.path.join(mask_dir, mask_file_same_name)
            
            if os.path.exists(mask_path_with_labels):
                self.pairs.append((img_file, mask_file_with_labels))
            elif os.path.exists(mask_path_same_name):
                self.pairs.append((img_file, mask_file_same_name))
            else:
                print(f"警告：找不到标签文件 {mask_file_with_labels} 或 {mask_file_same_name}")
        
        assert len(self.pairs) > 0, f"未找到任何匹配的图像和掩码文件"
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        hyperspectral_filename, mask_filename = self.pairs[idx]
        
        # 加载高光谱图像
        hyperspectral_data = scipy.io.loadmat(os.path.join(self.image_dir, hyperspectral_filename))
        hyperspectral_image = hyperspectral_data['mat']
        
        # 加载掩码图像
        mask_data = scipy.io.loadmat(os.path.join(self.mask_dir, mask_filename))
        mask_image = mask_data['mat']
        
        # 将mask限制在[0, 1]范围内（二分类）
        mask_image = np.clip(mask_image, 0, 1)
        
        # 归一化：min-max归一化到[0, 1]
        hyperspectral_image = hyperspectral_image.astype(np.float32)
        img_min = hyperspectral_image.min()
        img_max = hyperspectral_image.max()
        if img_max > img_min:
            hyperspectral_image = (hyperspectral_image - img_min) / (img_max - img_min)
        else:
            hyperspectral_image = hyperspectral_image * 0.0
        mask_image = mask_image.astype(np.float32)
        
        # 先转为 tensor，permute调整为 (C, H, W) 格式
        hyperspectral_image = torch.tensor(hyperspectral_image, dtype=torch.float32).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        mask_image = torch.tensor(mask_image, dtype=torch.float32)
        
        # 应用变换（如果有）- transform会接收tensor并返回tensor
        if self.transform:
            hyperspectral_image, mask_image = self.transform(hyperspectral_image, mask_image)
        
        # 获取图像基础名（无扩展名）
        image_name = os.path.splitext(hyperspectral_filename)[0]
        
        return hyperspectral_image, mask_image, image_name


# Patch数据集（用于patch训练）
class HyperspectralPatchDataset(Dataset):
    def __init__(self, patch_image_dir, patch_mask_dir, patch_list, transform=None):
        """
        Patch数据集，用于基于patch的训练
        
        Args:
            patch_image_dir: patch图像目录（例如：/data/CXY/gwj/Unet/patches）
            patch_mask_dir: patch掩膜目录（例如：/data/CXY/gwj/Unet/patches_label）
            patch_list: patch名称列表（例如：["032236-20x-roi4_01", "032236-20x-roi4_02", ...]）
            transform: 数据增强（翻转等）
        """
        self.patch_image_dir = patch_image_dir
        self.patch_mask_dir = patch_mask_dir
        self.patch_list = patch_list
        self.transform = transform
        
        print(f'Patch数据集: {len(self.patch_list)} 个patches')
    
    def __len__(self):
        return len(self.patch_list)
    
    def __getitem__(self, idx):
        # 获取patch名称
        patch_name = self.patch_list[idx]
        
        # 构建patch文件路径
        patch_image_path = os.path.join(self.patch_image_dir, f"{patch_name}.mat")
        patch_mask_path = os.path.join(self.patch_mask_dir, f"{patch_name}.mat")
        
        # 加载patch
        try:
            image_data = scipy.io.loadmat(patch_image_path)
            mask_data = scipy.io.loadmat(patch_mask_path)
        except Exception as e:
            print(f"错误：无法加载patch {patch_name}: {e}")
            raise
        
        # 获取数据（排除元数据）
        image_keys = [k for k in image_data.keys() if not k.startswith('__')]
        mask_keys = [k for k in mask_data.keys() if not k.startswith('__')]
        
        if not image_keys or not mask_keys:
            raise ValueError(f"Patch {patch_name} 没有有效数据")
        
        patch_image = image_data[image_keys[0]]
        patch_mask = mask_data[mask_keys[0]]
        
        # 将mask限制在[0, 1]范围内（二分类）
        patch_mask = np.clip(patch_mask, 0, 1)
        
        # 不需要归一化，patch在切分时已经是归一化后的数据
        # 直接转为tensor，permute调整为 (C, H, W) 格式
        patch_image = torch.tensor(patch_image, dtype=torch.float32).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        patch_mask = torch.tensor(patch_mask, dtype=torch.float32)
        
        # 应用变换（数据增强：翻转等）
        if self.transform:
            patch_image, patch_mask = self.transform(patch_image, patch_mask)
        
        return patch_image, patch_mask, patch_name
