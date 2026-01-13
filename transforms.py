import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    # 支持PIL Image和tensor
    if torch.is_tensor(img):
        # tensor: (C, H, W) 或 (H, W)
        if img.ndim == 3:
            _, h, w = img.shape
        else:
            h, w = img.shape
        min_size = min(h, w)
        if min_size < size:
            padh = size - h if h < size else 0
            padw = size - w if w < size else 0
            # F.pad for tensor: (left, right, top, bottom)
            img = F.pad(img, (0, padw, 0, padh), fill=fill)
    else:
        # PIL Image
        min_size = min(img.size)
        if min_size < size:
            ow, oh = img.size
            padh = size - oh if oh < size else 0
            padw = size - ow if ow < size else 0
            img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# 随机缩放
class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


# 随机水平翻转
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


# 随机垂直翻转
class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


# 随机裁剪
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


# 中心裁剪
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, [self.size, self.size])
        target = F.center_crop(target, [self.size, self.size])
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


# ============================================================================
# 高光谱特有数据增强
# ============================================================================

class SpectralMixTransform(object):
    """
    光谱混合增强：随机混合光谱通道
    通过线性混合不同通道，增强光谱多样性，防止网络记忆特定光谱特征
    """
    def __init__(self, min_channels=1, max_channels=4, mix_factor_range=(0.35, 0.65), prob=0.5):
        """
        Args:
            min_channels: 最少混合通道数
            max_channels: 最多混合通道数
            mix_factor_range: 混合系数范围 (alpha, 1-alpha)
            prob: 应用概率
        """
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.mix_factor_range = mix_factor_range
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        # 支持tensor和numpy
        is_tensor = torch.is_tensor(image)
        if is_tensor:
            data = image.clone()
            device = data.device
            dtype = data.dtype
        else:
            data = np.array(image)
            dtype = data.dtype
        
        # 需要至少2个通道
        num_channels = data.shape[0] if len(data.shape) >= 3 else 1
        if num_channels < 2:
            return image, target
        
        # 随机选择要混合的通道数
        max_ch = min(self.max_channels, num_channels - 1)
        if max_ch < self.min_channels:
            return image, target
        
        num_to_mix = random.randint(self.min_channels, max_ch)
        channel_indices = random.sample(range(num_channels), num_to_mix)
        
        # 对每个选中的通道，随机选择另一个通道进行混合
        for idx in channel_indices:
            partner_candidates = [c for c in range(num_channels) if c != idx]
            partner = random.choice(partner_candidates)
            mix_factor = random.uniform(*self.mix_factor_range)
            
            if is_tensor:
                data[idx] = mix_factor * data[idx] + (1.0 - mix_factor) * data[partner]
            else:
                data[idx] = mix_factor * data[idx] + (1.0 - mix_factor) * data[partner]
        
        if is_tensor:
            return data.to(device=device, dtype=dtype), target
        return data.astype(dtype), target


class RandomGaussianNoise(object):
    """添加高斯噪声增强"""
    def __init__(self, prob=0.5, std_range=(0.01, 0.05)):
        self.prob = prob
        self.std_range = std_range
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        is_tensor = torch.is_tensor(image)
        
        std = random.uniform(*self.std_range)
        
        if is_tensor:
            noise = torch.randn_like(image) * std
            image = image + noise
        else:
            noise = np.random.randn(*image.shape) * std
            image = image + noise
        
        return image, target


class RandomBrightness(object):
    """随机亮度调整"""
    def __init__(self, prob=0.5, brightness_range=(0.8, 1.2)):
        self.prob = prob
        self.brightness_range = brightness_range
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        factor = random.uniform(*self.brightness_range)
        
        if torch.is_tensor(image):
            image = image * factor
        else:
            image = image * factor
        
        return image, target


class RandomContrast(object):
    """随机对比度调整"""
    def __init__(self, prob=0.5, contrast_range=(0.8, 1.2)):
        self.prob = prob
        self.contrast_range = contrast_range
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        factor = random.uniform(*self.contrast_range)
        
        if torch.is_tensor(image):
            mean = image.mean()
            image = (image - mean) * factor + mean
        else:
            mean = image.mean()
            image = (image - mean) * factor + mean
        
        return image, target


class RandomRotation90(object):
    """随机90度旋转"""
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        # 随机选择旋转次数 (0, 1, 2, 3 对应 0°, 90°, 180°, 270°)
        k = random.randint(0, 3)
        
        if torch.is_tensor(image):
            # PyTorch tensor: (C, H, W)
            image = torch.rot90(image, k, dims=[-2, -1])
            target = torch.rot90(target, k, dims=[-2, -1]) if torch.is_tensor(target) else np.rot90(target, k)
        else:
            # NumPy array: (C, H, W) 
            image = np.rot90(image, k, axes=(-2, -1))
            target = np.rot90(target, k, axes=(-2, -1)) if isinstance(target, np.ndarray) else target
        
        return image, target


class RandomChannelDropout(object):
    """随机丢弃部分光谱通道"""
    def __init__(self, prob=0.5, dropout_prob=0.1):
        """
        Args:
            prob: 应用该增强的概率
            dropout_prob: 每个通道被dropout的概率
        """
        self.prob = prob
        self.dropout_prob = dropout_prob
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        
        is_tensor = torch.is_tensor(image)
        num_channels = image.shape[0] if len(image.shape) >= 3 else 1
        
        if num_channels < 2:
            return image, target
        
        # 随机决定每个通道是否dropout
        for c in range(num_channels):
            if random.random() < self.dropout_prob:
                if is_tensor:
                    image[c] = 0
                else:
                    image[c] = 0
        
        return image, target
