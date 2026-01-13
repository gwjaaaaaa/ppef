"""
Training Script for UNet_PPEF++
使用PPEF++框架训练高光谱图像分割模型

使用说明：
1. 确保所有PPEF++模块已正确安装
2. 修改数据路径和参数配置
3. 运行：python train_ppefpp.py --fold 0

主要改动：
- 使用UNet_PPEFPP模型（替代UNet）
- 使用PPEFPPLoss损失函数
- 在训练循环中获取unmixing输出
- 记录额外的损失项（重建、平滑、多样性、熵）
"""

import os
import sys
import torch
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# 添加父目录到Python路径，以便导入train.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 从train.py导入需要的类和函数
from train import (
    HyperspectralPresetTrain,
    EMA
)
from dataset import HyperspectralDatasetWithKeys, HyperspectralPatchDataset
from utils import calculate_metrics, sliding_window_predict

# 导入PPEF++模块
from model_ppefpp import UNet_PPEFPP
from loss_ppefpp import PPEFPPLoss


def train_one_epoch_ppefpp(model, criterion, optim, train_loader, device, scaler=None, epoch=None):
    """
    训练一个epoch（PPEF++版本）
    
    Args:
        epoch: 当前epoch（用于warmup）
    
    Returns:
        avg_loss: 平均总损失
        loss_details: 各项损失的平均值
        lr: 当前学习率
    """
    model.train()
    
    total_loss = 0.0
    loss_details = {}
    
    for batch_idx, (images, targets, _) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # 混合精度训练
        if scaler is not None:
            with autocast(device_type='cuda'):
                # 前向传播（直接返回unmixing输出）
                outputs, A2, X2_hat, X2_down = model(images, return_unmixing=True)
                prototypes = model.get_prototypes(to_cpu=False)  # ✅ 保持在GPU上
                
                # 计算损失（立即使用，用完即释放）
                loss, loss_dict = criterion(
                    outputs, targets,
                    A2=A2, X2_hat=X2_hat, X2_down=X2_down,
                    prototypes=prototypes,
                    epoch=epoch  # 传入epoch用于warmup
                )
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)  # ✅ 优化：清零梯度并释放内存
        else:
            # 前向传播（直接返回unmixing输出）
            outputs, A2, X2_hat, X2_down = model(images, return_unmixing=True)
            prototypes = model.get_prototypes(to_cpu=False)  # ✅ 保持在GPU上
            
            # 计算损失（立即使用，用完即释放）
            loss, loss_dict = criterion(
                outputs, targets,
                A2=A2, X2_hat=X2_hat, X2_down=X2_down,
                prototypes=prototypes
            )
            
            # 反向传播
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)  # ✅ 优化：清零梯度并释放内存
        
        total_loss += loss.item()
        
        # 累积各项损失（跳过非tensor值，如warmup标志）
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):  # 只处理tensor
                if key not in loss_details:
                    loss_details[key] = 0.0
                loss_details[key] += value.item()
    
    # 计算平均
    avg_loss = total_loss / len(train_loader)
    for key in loss_details:
        loss_details[key] /= len(train_loader)
    
    # 获取学习率
    lr = optim.param_groups[0]['lr']
    
    return avg_loss, loss_details, lr


def calculate_val_loss_and_dice_ppefpp_sliding(model, val_loader, device, patch_size=256, overlap=0.5):
    """
    使用滑窗预测计算验证集上的损失和Dice（PPEF++版本）
    
    用于patch训练模式：验证时使用完整图像，但通过滑窗方式预测避免显存爆炸
    
    Args:
        model: PPEF++模型
        val_loader: 验证集DataLoader（完整图像）
        device: 设备
        patch_size: 滑窗patch尺寸
        overlap: 重叠比例（0.5表示50%重叠）
    
    Returns:
        avg_val_loss: 验证损失
        avg_val_dice: 验证Dice
        loss_details: 各项损失的平均值
    """
    from utils import sliding_window_predict
    import torch.nn.functional as F
    
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    
    with torch.no_grad():
        for images, targets, _ in val_loader:
            # images: (B, C, H, W), targets: (B, H, W)
            B = images.shape[0]
            
            for i in range(B):
                img = images[i]  # (C, H, W)
                gt = targets[i].to(device).float()  # (H, W)
                
                # 滑窗预测（使用重叠滑窗+高斯权重融合）
                # 注意：sliding_window_predict会自动处理return_unmixing=False
                prob_map, pred_mask = sliding_window_predict(
                    model, img, device,
                    patch_size=patch_size,
                    overlap=overlap
                )
                # prob_map: (H, W) 概率图
                # pred_mask: (H, W) 预测mask {0, 1}
                
                # 计算Dice
                pred_flat = pred_mask.float()
                gt_flat = gt.float()
                
                smooth = 1e-5
                intersection = (pred_flat * gt_flat).sum()
                dice = (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
                
                total_dice += dice.item()
                n += 1
                
                # 计算Loss（只计算分割损失，不计算unmixing损失）
                # 使用prob_map直接计算BCE损失
                prob_map_expanded = prob_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                gt_expanded = gt.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                
                # BCE损失（注意：prob_map是sigmoid后的值）
                bce_val = F.binary_cross_entropy(prob_map_expanded, gt_expanded)
                
                # Dice损失
                dice_loss_val = 1 - dice
                
                # 总损失（BCE + Dice）
                loss = 0.5 * bce_val + 0.5 * dice_loss_val
                total_loss += loss.item()
    
    avg_val_loss = total_loss / max(n, 1)
    avg_val_dice = total_dice / max(n, 1)
    
    # 损失详情（只有分割损失）
    loss_details = {
        'seg': avg_val_loss
    }
    
    return avg_val_loss, avg_val_dice, loss_details


# ============================================================================
# 训练期间的可视化辅助函数 - 端元和专家权重演化追踪
# ============================================================================

def extract_endmember_info(model):
    """
    提取端元的关键信息（用于演化追踪）
    
    Returns:
        dict: {
            'spectra': (K, C) numpy数组 - 端元光谱,
            'correlation_matrix': (K, K) numpy数组 - 相关性矩阵,
            'orthogonality_score': float - 正交性指标（非对角线绝对值均值）
        }
    """
    import torch.nn.functional as F
    
    # 处理torch.compile()包装后的模型
    actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    # 提取端元光谱（注意：模型中的属性名是 unmix_head，不是 unmixing_head）
    P = actual_model.unmix_head.P_spec.detach().cpu()  # (K, num_bands)
    
    # 归一化
    P_norm = F.normalize(P, dim=1)
    
    # 计算相关矩阵
    corr_matrix = torch.matmul(P_norm, P_norm.T).numpy()  # (K, K)
    
    # 计算正交性指标（非对角线绝对值均值，越小越好）
    K = corr_matrix.shape[0]
    mask = ~np.eye(K, dtype=bool)
    off_diag = corr_matrix[mask]
    orthogonality_score = np.abs(off_diag).mean()
    
    return {
        'spectra': P.numpy(),  # (K, C)
        'correlation_matrix': corr_matrix,  # (K, K)
        'orthogonality_score': orthogonality_score
    }


def _entropy_norm(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    W: (N, E) 概率分布
    返回: (N,) 归一化熵 H/log(E)，范围[0,1]
    注意：熵必须按"每个样本"计算，再做均值/方差（不能用平均权重算熵）
    """
    E = W.shape[1]
    if E <= 1:
        return np.zeros((W.shape[0],), dtype=np.float32)
    W = np.clip(W, eps, 1.0)
    H = -np.sum(W * np.log(W), axis=1)      # (N,)
    return (H / np.log(E)).astype(np.float32)


def extract_expert_weights_on_monitor(model, monitor_loader, device):
    """
    固定 Monitor Set 上提取：
    - avg_weights[module] = (E,)
    - stats[module] = {
        entropy_mean, entropy_std,
        top1_rate (E,),
        w_std (E,)
      }
    
    Returns:
        tuple: (avg_weights, stats)
    """
    model.eval()
    
    # torch.compile() 兼容：用原始模型跑 forward，避免 attribute 保存异常
    actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    actual_model.eval()
    
    # 6个CSSE模块
    csse_modules = {
        'Enc_X2': actual_model.csse_enc1,
        'Enc_X3': actual_model.csse_enc2,
        'Enc_X4': actual_model.csse_enc3,
        'Dec_Up1': actual_model.csse_dec1,
        'Dec_Up2': actual_model.csse_dec2,
        'Dec_Up3': actual_model.csse_dec3,
    }
    
    # 默认所有层是同样的专家数（当前配置：2个专家），这里做强一致性校验，防止“某层配置跑偏”
    num_experts = list(csse_modules.values())[0].num_active_experts
    for k, m in csse_modules.items():
        if m.num_active_experts != num_experts:
            raise RuntimeError(f"[CSSE] num_active_experts mismatch at {k}: {m.num_active_experts} vs {num_experts}")
    
    
    per_module_W = {name: [] for name in csse_modules.keys()}
    
    with torch.inference_mode():
        for images, _, _ in monitor_loader:
            img = images.to(device, non_blocking=True)  # (1, C, H, W)
            
            # forward 一次，让各 CSSE router 在 eval 下写入 last_weights
            _ = actual_model(img, return_unmixing=False)
            
            for name, csse_module in csse_modules.items():
                if csse_module.router is None:
                    # 只有一个专家：one-hot
                    w = np.zeros(num_experts, dtype=np.float32)
                    w[0] = 1.0
                    per_module_W[name].append(w)
                    continue
                
                lw = csse_module.router.last_weights
                if lw is None:
                    # fallback：均匀（尽量不要频繁出现）
                    w = np.full(num_experts, 1.0 / num_experts, dtype=np.float32)
                    per_module_W[name].append(w)
                    continue
                
                if isinstance(lw, np.ndarray):
                    w = lw[0].astype(np.float32)  # (E,)
                else:
                    w = lw[0].detach().float().cpu().numpy().astype(np.float32)
                
                # 强校验：维度 + 归一
                if w.shape[0] != num_experts:
                    raise RuntimeError(f"[{name}] weight dim mismatch: got {w.shape[0]} vs expected {num_experts}")
                s = float(w.sum())
                if abs(s - 1.0) > 1e-3:
                    raise RuntimeError(f"[{name}] weights not normalized: sum={s:.6f}")
                
                per_module_W[name].append(w)
    
    # 汇总 avg + stats
    avg_weights = {}
    stats = {}
    
    for name, w_list in per_module_W.items():
        W = np.stack(w_list, axis=0)          # (N,E)
        avg_weights[name] = W.mean(axis=0)    # (E,)
        
        # 熵：按样本算 -> 均值/方差
        Hn = _entropy_norm(W)                 # (N,)
        # top1：按样本argmax统计频率
        top1 = W.argmax(axis=1)               # (N,)
        top1_rate = np.bincount(top1, minlength=num_experts) / len(top1)
        
        stats[name] = {
            "entropy_mean": float(Hn.mean()),
            "entropy_std":  float(Hn.std()),
            "top1_rate":    top1_rate.astype(float).tolist(),
            "w_std":        W.std(axis=0).astype(float).tolist()
        }
    
    return avg_weights, stats


def plot_endmember_evolution(evolution_data, save_path, wavelengths=None):
    """
    绘制端元演化图
    
    Args:
        evolution_data: list of dict，每个dict包含一个epoch的端元信息
        save_path: 保存路径
        wavelengths: 波长数组（可选）
    """
    epochs = [d['epoch'] for d in evolution_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 正交性指标演化
    ax = axes[0, 0]
    orth_scores = [d['orthogonality_score'] for d in evolution_data]
    ax.plot(epochs, orth_scores, marker='o', linewidth=2, color='#E74C3C')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Orthogonality Score\n(lower is better)', fontsize=11)
    ax.set_title('Endmember Orthogonality Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good (<0.1)')
    ax.legend()
    
    # 2. 端元光谱演化（显示初始和最终）
    ax = axes[0, 1]
    K = evolution_data[0]['spectra'].shape[0]
    num_bands = evolution_data[0]['spectra'].shape[1]
    x_axis = wavelengths if wavelengths is not None else np.arange(num_bands)
    xlabel = 'Wavelength (nm)' if wavelengths is not None else 'Band Index'
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    # 显示初始状态（虚线）
    for k in range(K):
        ax.plot(x_axis, evolution_data[0]['spectra'][k], 
                linestyle='--', color=colors[k], alpha=0.3, label=f'E{k+1} (Init)')
    # 显示最终状态（实线）
    for k in range(K):
        ax.plot(x_axis, evolution_data[-1]['spectra'][k], 
                linestyle='-', color=colors[k], linewidth=2, label=f'E{k+1} (Final)')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Reflectance', fontsize=11)
    ax.set_title('Endmember Spectra: Initial vs Final', fontsize=12, fontweight='bold')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. 相关性矩阵演化（显示初始）
    ax = axes[1, 0]
    corr_init = evolution_data[0]['correlation_matrix']
    im = ax.imshow(corr_init, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'Correlation Matrix (Epoch {epochs[0]})', fontsize=11, fontweight='bold')
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f'E{i+1}' for i in range(K)])
    ax.set_yticklabels([f'E{i+1}' for i in range(K)])
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 4. 相关性矩阵演化（显示最终）
    ax = axes[1, 1]
    corr_final = evolution_data[-1]['correlation_matrix']
    im = ax.imshow(corr_final, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'Correlation Matrix (Epoch {epochs[-1]})', fontsize=11, fontweight='bold')
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f'E{i+1}' for i in range(K)])
    ax.set_yticklabels([f'E{i+1}' for i in range(K)])
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Endmember Evolution During Training', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 端元演化图已保存: {save_path}")


def plot_expert_weights_evolution(evolution_data, save_path):
    """
    绘制专家权重演化图
    
    Args:
        evolution_data: list of dict，每个dict包含一个epoch的专家权重
        save_path: 保存路径
    """
    epochs = [d['epoch'] for d in evolution_data]
    
    # 动态专家系统（根据实际启用的专家数量）
    if len(evolution_data) > 0 and 'expert_weights' in evolution_data[0]:
        first_weights = list(evolution_data[0]['expert_weights'].values())[0]
        num_experts = len(first_weights)
    else:
        num_experts = 2  # 默认2专家
    
    # 当前配置：专家1+2（禁用了专家3和4）
    expert_names = ['Expert 1 (Spectral-Local)', 'Expert 2 (Spectral-Global)']
    colors = ['#FF6B6B', '#4ECDC4']
    
    # 6个模块
    module_names = ['Enc_X2', 'Enc_X3', 'Enc_X4', 'Dec_Up1', 'Dec_Up2', 'Dec_Up3']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, module_name in enumerate(module_names):
        ax = axes[i]
        
        # 提取该模块的权重演化
        weights_over_time = []
        for d in evolution_data:
            if module_name in d['expert_weights']:
                weights_over_time.append(d['expert_weights'][module_name])
        
        if len(weights_over_time) == 0:
            continue
        
        weights_array = np.array(weights_over_time)  # (num_epochs, num_experts)
        
        # 动态绘制曲线（根据实际专家数量）
        for j in range(min(weights_array.shape[1], len(expert_names))):
            ax.plot(epochs[:len(weights_over_time)], weights_array[:, j], 
                   marker='o', label=expert_names[j], color=colors[j], linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Expert Weight', fontsize=11)
        ax.set_title(f'{module_name}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        uniform_weight = 1.0 / num_experts
        ax.axhline(y=uniform_weight, color='gray', linestyle='--', alpha=0.3, label=f'Uniform ({uniform_weight:.2f})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Expert Weights Evolution During Training', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 专家权重演化图已保存: {save_path}")


def save_endmember_data(evolution_data, save_dir):
    """
    保存端元演化数据到CSV和JSON文件
    
    Args:
        evolution_data: list of dict，每个dict包含一个epoch的端元信息
        save_dir: 保存目录
    """
    # 1. 保存端元正交性演化 (CSV)
    orth_data = {
        'epoch': [d['epoch'] for d in evolution_data],
        'orthogonality_score': [d['orthogonality_score'] for d in evolution_data]
    }
    df_orth = pd.DataFrame(orth_data)
    orth_path = f'{save_dir}/endmember_orthogonality.csv'
    df_orth.to_csv(orth_path, index=False)
    print(f"✓ 端元正交性数据已保存: {orth_path}")
    
    # 2. 保存端元光谱（初始 vs 最终）(CSV)
    # 格式：endmember, band_idx, epoch, reflectance
    spectra_records = []
    K = evolution_data[0]['spectra'].shape[0]  # 端元数量
    num_bands = evolution_data[0]['spectra'].shape[1]  # 波段数
    
    # 初始光谱
    for k in range(K):
        for band_idx in range(num_bands):
            spectra_records.append({
                'endmember': f'E{k+1}',
                'band_idx': band_idx,
                'epoch': evolution_data[0]['epoch'],
                'reflectance': evolution_data[0]['spectra'][k, band_idx]
            })
    
    # 最终光谱
    for k in range(K):
        for band_idx in range(num_bands):
            spectra_records.append({
                'endmember': f'E{k+1}',
                'band_idx': band_idx,
                'epoch': evolution_data[-1]['epoch'],
                'reflectance': evolution_data[-1]['spectra'][k, band_idx]
            })
    
    df_spectra = pd.DataFrame(spectra_records)
    spectra_path = f'{save_dir}/endmember_spectra_init_final.csv'
    df_spectra.to_csv(spectra_path, index=False)
    print(f"✓ 端元光谱数据已保存: {spectra_path}")
    
    # 3. 保存端元相关性矩阵（初始 vs 最终）(JSON)
    corr_data = {
        f"epoch_{evolution_data[0]['epoch']}": {},
        f"epoch_{evolution_data[-1]['epoch']}": {}
    }
    
    # 初始相关性矩阵
    corr_init = evolution_data[0]['correlation_matrix']
    for i in range(K):
        corr_data[f"epoch_{evolution_data[0]['epoch']}"][f'E{i+1}'] = corr_init[i].tolist()
    
    # 最终相关性矩阵
    corr_final = evolution_data[-1]['correlation_matrix']
    for i in range(K):
        corr_data[f"epoch_{evolution_data[-1]['epoch']}"][f'E{i+1}'] = corr_final[i].tolist()
    
    corr_path = f'{save_dir}/endmember_correlation_init_final.json'
    with open(corr_path, 'w') as f:
        json.dump(corr_data, f, indent=2)
    print(f"✓ 端元相关性矩阵已保存: {corr_path}")
    
    # 4. 【可选】保存完整的端元演化（所有epoch）(NPY)
    # 如果记录的epoch较多，可以保存完整的演化过程
    if len(evolution_data) > 2:
        spectra_full = np.array([d['spectra'] for d in evolution_data])  # (num_epochs, K, num_bands)
        epochs_full = np.array([d['epoch'] for d in evolution_data])
        
        full_path = f'{save_dir}/endmember_spectra_full.npz'
        np.savez(full_path, 
                 spectra=spectra_full, 
                 epochs=epochs_full,
                 orthogonality_scores=np.array([d['orthogonality_score'] for d in evolution_data]))
        print(f"✓ 完整端元演化数据已保存: {full_path}")


def save_expert_weights_data(evolution_data, save_dir):
    """
    保存专家权重演化数据到CSV文件（动态适配专家数量）
    
    Args:
        evolution_data: list of dict，每个dict包含一个epoch的专家权重
        save_dir: 保存目录
    """
    # 提取数据并转换为CSV格式
    # 格式：epoch, module, expert1_spectral_local, expert2_spectral_global, expert3_spatial_edge
    records = []
    
    module_names = ['Enc_X2', 'Enc_X3', 'Enc_X4', 'Dec_Up1', 'Dec_Up2', 'Dec_Up3']
    # 当前配置：专家1+2（expert3和expert4已禁用）
    # 权重数组索引0对应expert1，索引1对应expert2
    expert_col_names = ['expert1_spectral_local', 'expert2_spectral_global', 'expert3_spatial_edge']
    
    for data in evolution_data:
        epoch = data['epoch']
        weights_dict = data['expert_weights']
        
        for module_name in module_names:
            if module_name in weights_dict:
                weights = weights_dict[module_name]  # (num_active_experts,) numpy array
                record = {
                    'epoch': epoch,
                    'module': module_name,
                }
                # 动态添加专家权重列（根据启用的专家映射）
                # 当前配置：weights[0]=expert1, weights[1]=expert2
                active_experts = ['expert1_spectral_local', 'expert2_spectral_global']
                for i in range(len(weights)):
                    if i < len(active_experts):
                        record[active_experts[i]] = weights[i]
                records.append(record)
    
    df_weights = pd.DataFrame(records)
    weights_path = f'{save_dir}/expert_weights_evolution.csv'
    df_weights.to_csv(weights_path, index=False)
    print(f"✓ 专家权重演化数据已保存: {weights_path}")
    
    # 【可选】也保存JSON格式（更结构化，动态适配专家数量）
    json_data = {}
    
    # 动态初始化专家字段（根据第一个epoch的实际专家数量）
    if len(evolution_data) > 0:
        first_weights_dict = evolution_data[0]['expert_weights']
        first_module = module_names[0]
        if first_module in first_weights_dict:
            num_experts = len(first_weights_dict[first_module])
            # 当前配置：只有expert1和expert2启用
            expert_keys = ['expert1_spectral_local', 'expert2_spectral_global'][:num_experts]
        else:
            expert_keys = expert_col_names  # 回退到默认
    else:
        expert_keys = expert_col_names  # 回退到默认
    
    # 初始化JSON结构
    for module_name in module_names:
        json_data[module_name] = {'epochs': []}
        for key in expert_keys:
            json_data[module_name][key] = []
    
    # 填充数据
    for data in evolution_data:
        epoch = data['epoch']
        weights_dict = data['expert_weights']
        
        for module_name in module_names:
            if module_name in weights_dict:
                weights = weights_dict[module_name]
                json_data[module_name]['epochs'].append(epoch)
                # 动态添加权重值
                for i, key in enumerate(expert_keys):
                    if i < len(weights):
                        json_data[module_name][key].append(float(weights[i]))
    
    json_path = f'{save_dir}/expert_weights_evolution.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ 专家权重演化数据已保存（JSON格式）: {json_path}")


def save_expert_stats_data(evolution_data, save_dir):
    """
    输出 3 个CSV：
      1) expert_entropy_evolution.csv: epoch,module,entropy_mean,entropy_std,val_dice
      2) expert_top1_rate_evolution.csv: epoch,module,top1_rate_e1..eE
      3) expert_weight_std_evolution.csv: epoch,module,std_e1..eE
    """
    module_names = ['Enc_X2', 'Enc_X3', 'Enc_X4', 'Dec_Up1', 'Dec_Up2', 'Dec_Up3']
    
    entropy_records = []
    top1_records = []
    std_records = []
    
    for d in evolution_data:
        epoch = d.get('epoch', None)
        stats = d.get('expert_stats', None)
        if epoch is None or stats is None:
            continue
        
        val_dice = d.get('val_dice', None)
        
        for m in module_names:
            if m not in stats:
                continue
            
            s = stats[m]
            entropy_records.append({
                'epoch': epoch,
                'module': m,
                'entropy_mean': s['entropy_mean'],
                'entropy_std':  s['entropy_std'],
                'val_dice':     val_dice
            })
            
            # 动态专家数（从top1_rate长度推断）
            top1_rate = s['top1_rate']
            w_std = s['w_std']
            E = min(len(top1_rate), len(w_std))
            
            r_top1 = {'epoch': epoch, 'module': m}
            r_std  = {'epoch': epoch, 'module': m}
            for i in range(E):
                r_top1[f'top1_rate_e{i+1}'] = float(top1_rate[i])
                r_std[f'std_e{i+1}'] = float(w_std[i])
            
            top1_records.append(r_top1)
            std_records.append(r_std)
    
    entropy_path = os.path.join(save_dir, 'expert_entropy_evolution.csv')
    top1_path    = os.path.join(save_dir, 'expert_top1_rate_evolution.csv')
    std_path     = os.path.join(save_dir, 'expert_weight_std_evolution.csv')
    
    pd.DataFrame(entropy_records).to_csv(entropy_path, index=False)
    pd.DataFrame(top1_records).to_csv(top1_path, index=False)
    pd.DataFrame(std_records).to_csv(std_path, index=False)
    
    print(f"✓ Expert stats saved:")
    print(f"  - {entropy_path}")
    print(f"  - {top1_path}")
    print(f"  - {std_path}")


def plot_expert_entropy_evolution(evolution_data, save_path):
    """绘制专家路由熵演化图"""
    module_names = ['Enc_X2', 'Enc_X3', 'Enc_X4', 'Dec_Up1', 'Dec_Up2', 'Dec_Up3']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, m in enumerate(module_names):
        ax = axes[i]
        ep, meanH, stdH = [], [], []
        
        for d in evolution_data:
            stats = d.get('expert_stats', None)
            if stats is None or m not in stats:
                continue
            ep.append(d['epoch'])
            meanH.append(stats[m]['entropy_mean'])
            stdH.append(stats[m]['entropy_std'])
        
        if len(ep) == 0:
            continue
        
        meanH = np.array(meanH); stdH = np.array(stdH)
        ax.plot(ep, meanH, marker='o', linewidth=2, color='#E74C3C')
        ax.fill_between(ep, meanH - stdH, meanH + stdH, alpha=0.2, color='#E74C3C')
        
        ax.set_title(f'{m} | Routing Entropy (norm)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('H(w)/log(E)', fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Routing Entropy Evolution (Monitor Set)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_expert_top1_evolution(evolution_data, save_path):
    """绘制Top-1专家选择频率演化图"""
    module_names = ['Enc_X2', 'Enc_X3', 'Enc_X4', 'Dec_Up1', 'Dec_Up2', 'Dec_Up3']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    expert_labels = ['Expert 1 (Spectral-Local)', 'Expert 2 (Spectral-Global)', 'Expert 3 (Spatial-Edge)']
    
    for i, m in enumerate(module_names):
        ax = axes[i]
        ep = []
        rates = []
        
        for d in evolution_data:
            stats = d.get('expert_stats', None)
            if stats is None or m not in stats:
                continue
            ep.append(d['epoch'])
            rates.append(stats[m]['top1_rate'])
        
        if len(ep) == 0:
            continue
        
        # rates: list of (E,) -> (T,E)
        rates = np.array(rates, dtype=np.float32)
        E = rates.shape[1]
        
        for e in range(E):
            label = expert_labels[e] if e < len(expert_labels) else f'Expert {e+1}'
            color = colors[e] if e < len(colors) else None
            ax.plot(ep, rates[:, e], marker='o', linewidth=2, label=label, color=color)
        
        ax.set_title(f'{m} | Top-1 Frequency', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Top-1 rate', fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle('Top-1 Expert Selection Frequency (Monitor Set)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_entropy_vs_valdice(evolution_data, save_path):
    """
    每个模块：entropy_mean 与 val_dice 同图对照（双y轴）
    """
    module_names = ['Enc_X2', 'Enc_X3', 'Enc_X4', 'Dec_Up1', 'Dec_Up2', 'Dec_Up3']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, m in enumerate(module_names):
        ax = axes[i]
        ep, ent, dice = [], [], []
        
        for d in evolution_data:
            stats = d.get('expert_stats', None)
            if stats is None or m not in stats:
                continue
            if 'val_dice' not in d:
                continue
            
            ep.append(d['epoch'])
            ent.append(stats[m]['entropy_mean'])
            dice.append(d['val_dice'])
        
        if len(ep) == 0:
            continue
        
        # 左轴：entropy
        line1 = ax.plot(ep, ent, marker='o', linewidth=2, color='#E74C3C', label='Entropy')
        ax.set_title(f'{m} | Entropy vs Val Dice', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Entropy (H/logE)', fontsize=11, color='#E74C3C')
        ax.tick_params(axis='y', labelcolor='#E74C3C')
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        
        # 右轴：val_dice
        ax2 = ax.twinx()
        line2 = ax2.plot(ep, dice, marker='s', linewidth=2, linestyle='--', color='#3498DB', label='Val Dice')
        ax2.set_ylabel('Val Dice', fontsize=11, color='#3498DB')
        ax2.tick_params(axis='y', labelcolor='#3498DB')
        ax2.set_ylim(0.0, 1.0)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower right', fontsize=9)
    
    plt.suptitle('Routing Entropy vs Val Dice (Monitor checkpoints)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main(args):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device training.")
    
    # 创建结果目录
    work_dir = '/data/CXY/gwj/WUnet/2DIM/appefnewfull'
    results_dir = f'{work_dir}/run_results_ppefpp'
    fold_dir = f'{results_dir}/fold_{args.fold}'
    os.makedirs(fold_dir, exist_ok=True)
    
    print(f"Fold {args.fold} 结果将保存到: {fold_dir}\n")
    
    # 保存配置
    with open(f'{fold_dir}/training_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 数据增强
    if args.patch_size is not None:
        train_tf = HyperspectralPresetTrain(
            rotation_prob=0.5,
            channel_dropout_prob=0.3,
            noise_prob=0.3,
            brightness_prob=0.3
        )
    else:
        train_tf = HyperspectralPresetTrain()
    
    # 加载数据划分
    splits_file = '/data/CXY/gwj/WUnet/2DIM/splits_im_patch.json' if args.patch_size else '/data/CXY/gwj/WUnet/2DIM/splits_im.json'
    
    with open(splits_file, 'r') as f:
        splits_data = json.load(f)
    
    if isinstance(splits_data, dict) and 'splits' in splits_data:
        splits_list = splits_data['splits']
    else:
        splits_list = splits_data
    
    train_keys = splits_list[args.fold]['train']
    val_keys = splits_list[args.fold]['val']
    
    # 创建数据集
    image_dir = '/home/ubuntu/dataset_Med/PLGC/IM/IM_HSI_mat'
    mask_dir = '/home/ubuntu/dataset_Med/PLGC/IM/IM_label_mat'
    
    if args.patch_size is not None:
        patch_image_dir = '/data/CXY/gwj/WUnet/2DIM/patches_im'
        patch_mask_dir = '/data/CXY/gwj/WUnet/2DIM/patches_im_label'
        
        trainDataset = HyperspectralPatchDataset(
            patch_image_dir=patch_image_dir,
            patch_mask_dir=patch_mask_dir,
            patch_list=train_keys,
            transform=train_tf
        )
        
        valDataset = HyperspectralDatasetWithKeys(
            image_dir=image_dir,
            mask_dir=mask_dir,
            keys=val_keys
        )
    else:
        trainDataset = HyperspectralDatasetWithKeys(
            image_dir=image_dir,
            mask_dir=mask_dir,
            keys=train_keys,
            transform=train_tf
        )
        
        valDataset = HyperspectralDatasetWithKeys(
            image_dir=image_dir,
            mask_dir=mask_dir,
            keys=val_keys
        )
    
    # DataLoader
    trainLoader = DataLoader(
        trainDataset, 
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    # 验证集batch_size：滑窗预测时可以使用batch_size>1来提升DataLoader效率
    # 虽然滑窗预测逐张处理，但batch_size>1可以让DataLoader并行加载多张图像
    val_batch_size = 2 if args.patch_size is not None else args.batch_size
    
    # 验证集DataLoader
    valLoader = DataLoader(
        valDataset,
        batch_size=val_batch_size,  # ✅ 验证时用batch_size=2（提升DataLoader效率）
        num_workers=4,  # 与训练集保持一致
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4  # 与训练集保持一致
    )
    
    # ============================================================================
    # ✅ 固定 Monitor Set（用于专家权重/丰度等可解释性可视化，跨epoch严格可比）
    # ============================================================================
    monitor_n = min(8, len(val_keys))  # 可以改成 5/8/10
    rng = np.random.default_rng(42 + args.fold)  # 固定种子，fold间也区分一下
    monitor_keys = rng.choice(val_keys, size=monitor_n, replace=False).tolist()
    
    # 可选：排序让输出更稳定（不影响"固定集合"，只影响顺序）
    monitor_keys = sorted(monitor_keys)
    
    # 保存下来，保证复现与论文一致
    with open(f'{fold_dir}/monitor_keys.json', 'w') as f:
        json.dump(monitor_keys, f, indent=2)
    
    monitorDataset = HyperspectralDatasetWithKeys(
        image_dir=image_dir,
        mask_dir=mask_dir,
        keys=monitor_keys
    )
    
    monitorLoader = DataLoader(
        monitorDataset,
        batch_size=1,      # ✅ 强烈建议1，简单、稳定、不会出现batch内选择偏差
        num_workers=2,     # 可按机器调，0/2都行
        shuffle=False,     # ✅ 固定顺序
        pin_memory=True
    )
    
    print(f"[Monitor Set] Using {len(monitor_keys)} fixed val samples for interpretability tracking.")
    
    print(f"\n{'='*70}")
    print(f"Dataset & DataLoader:")
    if args.patch_size is not None:
        print(f"  训练模式: PATCH训练 ({args.patch_size}×{args.patch_size})")
        print(f"  - Train: {len(trainDataset)} patches, batch_size={args.batch_size}")
        print(f"  - Val: {len(valDataset)} 张完整图像, batch_size={val_batch_size}")
        print(f"  验证模式: 滑窗预测 (patch_size={args.patch_size}, overlap=0.5)")
        print(f"  DataLoader: num_workers=4, prefetch_factor=4")
    else:
        print(f"  训练模式: 完整图像训练（不支持PPEF++）")
        print(f"  Train: {len(trainDataset)} samples")
        print(f"  Val: {len(valDataset)} samples")
    print(f"{'='*70}\n")
    
    # 创建模型（PPEF++）
    print(f"\n{'='*70}")
    print(f"Initializing UNet_PPEF++ model...")
    print(f"{'='*70}\n")
    
    model = UNet_PPEFPP(
        in_channels=40,
        out_channels=1,
        num_prototypes=args.num_prototypes,
        dropout_rate=args.dropout_rate,
        use_spgapp=True,
        use_csse=True,
        use_pgacpp=True
    )
    model.to(device)
    
    # ✅ 使用 torch.compile() 加速（PyTorch 2.0+）
    if args.use_compile:
        try:
            print(f"\n{'='*70}")
            print(f"🚀 启用 torch.compile() 加速...")
            print(f"   模式: reduce-overhead")
            print(f"   首次运行会进行编译（需要 10-30 秒），之后训练速度提升 15-25%")
            print(f"{'='*70}\n")
            model = torch.compile(model, mode='reduce-overhead')
            print(f"✅ 模型编译配置完成！\n")
        except Exception as e:
            print(f"⚠️  torch.compile() 编译失败，使用原始模型: {e}\n")
    else:
        print(f"ℹ️  未启用 torch.compile()（可通过 --use-compile 启用）\n")
    
    # 创建损失函数（PPEF++）
    criterion = PPEFPPLoss(
        lambda_recon=args.lambda_recon,
        lambda_smooth=args.lambda_smooth,
        lambda_div=args.lambda_div,
        lambda_entropy=args.lambda_entropy,
        lambda_orth=args.lambda_orth  # 【新增】原型正交约束
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度
    lf = lambda x: ((1 + np.cos(x * np.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # 混合精度
    scaler = GradScaler() if args.amp else None
    if args.amp:
        print(f"✓ 混合精度训练已启用 (FP16)\n")
    
    # EMA追踪器
    ema_tracker = EMA(decay=0.90)
    
    # 训练指标
    best_ema_dice = 0.0
    train_loss_list = []
    train_seg_loss_list = []  # ✅ 新增：记录训练时的分割损失（不含正则化）
    val_loss_list = []
    pseudo_dice_list = []
    ema_dice_list = []
    lr_list = []
    
    # 额外的PPEF++损失记录
    recon_loss_list = []
    smooth_loss_list = []
    div_loss_list = []
    entropy_loss_list = []
    orth_loss_list = []  # 【新增】原型正交约束损失
    
    # 【新增】端元和专家权重演化追踪
    endmember_evolution = []  # 存储端元演化数据
    expert_weights_evolution = []  # 存储专家权重演化数据
    
    print(f"\n{'='*70}")
    print(f"{'开始训练 UNet_PPEF++':^70}")
    print(f"{'='*70}")
    
    # ✅ torch.compile() 提示
    if args.use_compile:
        print(f"\n💡 提示：由于启用了 torch.compile()，")
        print(f"   第一个 epoch 会进行模型编译（可能需要额外 10-30 秒）")
        print(f"   之后的 epoch 将获得 15-25% 的速度提升 🚀\n")
    
    print()
    
    # 新加内容
    # ===== 训练前 baseline 记录（epoch=0）=====
    print("  --> 记录端元和专家权重演化 (Epoch 0 | before training)...")

    endmember_info = extract_endmember_info(model)
    endmember_info['epoch'] = 0
    endmember_evolution.append(endmember_info)

    avg_w, stat_w = extract_expert_weights_on_monitor(model, monitorLoader, device)
    expert_weights_evolution.append({
        'epoch': 0,
        'expert_weights': avg_w,
        'expert_stats': stat_w,
        'val_dice': None,   # 或者 0.0；反正 epoch0 没有 val
    })
    
    # 调试输出：显示初始专家权重
    if len(avg_w) > 0:
        first_module = list(avg_w.keys())[0]
        weights = avg_w[first_module]
        print(f"      初始专家权重 ({first_module}): {weights}")

    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_details, lr = train_one_epoch_ppefpp(
            model, criterion, optimizer, trainLoader, device, scaler, epoch=epoch
        )
        
        scheduler.step()
        
        # 验证（使用滑窗预测）
        if args.patch_size is not None:
            # Patch训练模式：使用滑窗预测（50%重叠+高斯权重融合）
            val_loss, val_dice, val_details = calculate_val_loss_and_dice_ppefpp_sliding(
                model, valLoader, device,
                patch_size=args.patch_size,
                overlap=0.5  # 50%重叠滑窗
            )
        else:
            # 完整图像训练模式：直接前向传播（不支持，建议使用patch模式）
            raise NotImplementedError("完整图像训练模式暂不支持PPEF++，请使用--patch-size参数启用patch训练")
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        # EMA Dice
        ema_dice = ema_tracker.update('dice', val_dice)
        
        # 记录指标
        train_loss_list.append(train_loss)
        train_seg_loss_list.append(train_details.get('seg', train_loss))  # ✅ 新增：记录训练分割损失
        val_loss_list.append(val_loss)
        pseudo_dice_list.append(val_dice)
        ema_dice_list.append(ema_dice)
        lr_list.append(lr)
        
        # 记录PPEF++特有的损失
        recon_loss_list.append(train_details.get('recon', 0.0))
        smooth_loss_list.append(train_details.get('smooth', 0.0))
        div_loss_list.append(train_details.get('div', 0.0))
        entropy_loss_list.append(train_details.get('entropy', 0.0))
        orth_loss_list.append(train_details.get('orth', 0.0))  # 【新增】
        
        # 打印训练信息（包含时间）
        train_seg_loss = train_details.get('seg', train_loss)
        log_msg = (f"Epoch {epoch+1:4d}/{args.epochs} | "
                  f"train_loss: {train_loss:.4f} (seg: {train_seg_loss:.4f}) | "
                  f"val_loss: {val_loss:.4f} | "
                  f"Dice: {val_dice:.4f} | EMA: {ema_dice:.4f} | lr: {lr:.2e} | "
                  f"{epoch_time:.1f}s")
        
        # 添加PPEF++损失信息
        if 'recon' in train_details:
            log_msg += f" | recon: {train_details['recon']:.4f}"
        
        print(log_msg, flush=True)
        
        # 保存最佳模型
        if ema_dice > best_ema_dice:
            best_ema_dice = ema_dice
            torch.save(model.state_dict(), f'{fold_dir}/model_best.pth')
            print(f"  --> New best EMA dice: {ema_dice:.4f}")
        
        # 写入日志（移除flush以减少I/O开销，系统会自动缓冲）
        with open(f'{fold_dir}/train_log.txt', "a") as f:
            f.write(log_msg + "\n")
        
        # 【新增】每隔10个epoch记录端元和专家权重演化
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f"  --> 记录端元和专家权重演化 (Epoch {epoch+1})...")
            
            # 提取端元信息
            endmember_info = extract_endmember_info(model)
            endmember_info['epoch'] = epoch + 1
            endmember_evolution.append(endmember_info)
            
            # 提取专家权重（在固定Monitor Set上）
            avg_w, stat_w = extract_expert_weights_on_monitor(model, monitorLoader, device)
            expert_weights_data = {
                'epoch': epoch + 1,
                'expert_weights': avg_w,
                'expert_stats': stat_w,
                'val_dice': float(val_dice),   # ✅ entropy vs val_dice 对照用
            }
            expert_weights_evolution.append(expert_weights_data)
            
            # 调试输出：显示专家权重
            if len(avg_w) > 0:
                first_module = list(avg_w.keys())[0]
                weights = avg_w[first_module]
                print(f"      专家权重 ({first_module}): {weights}")
            
            print(f"      端元正交性: {endmember_info['orthogonality_score']:.4f}")
        
        # 定期可视化（降低频率和DPI以节省时间）
        if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
            # 保存训练曲线
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Loss曲线（三条：总训练loss、训练分割loss、验证loss）
            axes[0, 0].plot(train_loss_list, label='Train Loss (Total)', color='#1f77b4', linewidth=2)
            axes[0, 0].plot(train_seg_loss_list, label='Train Loss (Seg Only)', color='#ff7f0e', linewidth=2, linestyle='--')
            axes[0, 0].plot(val_loss_list, label='Val Loss (Seg Only)', color='#2ca02c', linewidth=2)
            axes[0, 0].set_title('Loss Curves (Train Total vs Seg vs Val)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend(loc='upper right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Dice曲线
            axes[0, 1].plot(pseudo_dice_list, label='Pseudo Dice', alpha=0.6)
            axes[0, 1].plot(ema_dice_list, label='EMA Dice', linewidth=2)
            axes[0, 1].set_title('Dice Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 学习率
            axes[0, 2].plot(lr_list, color='orange')
            axes[0, 2].set_title('Learning Rate')
            axes[0, 2].grid(True, alpha=0.3)
            
            # PPEF++特有损失
            axes[1, 0].plot(recon_loss_list, label='Recon Loss')
            axes[1, 0].set_title('Spectral Reconstruction Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(smooth_loss_list, label='Smooth Loss')
            axes[1, 1].plot(entropy_loss_list, label='Entropy Loss')
            axes[1, 1].set_title('Regularization Losses')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].plot(div_loss_list, label='Diversity Loss')
            axes[1, 2].plot(orth_loss_list, label='Orthogonality Loss')  # 【新增】
            axes[1, 2].set_title('Prototype Losses')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{fold_dir}/training_progress.png', dpi=100)  # 降低DPI以加快保存
            plt.close()
            
            # 立即释放内存
            import gc
            gc.collect()
    
    # 保存最终模型
    torch.save(model.state_dict(), f'{fold_dir}/model_final.pth')
    
    # 【新增】绘制端元和专家权重演化图
    print(f"\n{'='*70}")
    print(f"生成端元和专家权重演化可视化...")
    print(f"{'='*70}\n")
    
    if len(endmember_evolution) > 0:
        # 绘制端元演化图
        plot_endmember_evolution(
            endmember_evolution,
            save_path=f'{fold_dir}/endmember_evolution.png',
            wavelengths=None  # 如果有波长文件，可以加载：np.loadtxt('wavelengths.txt')
        )
        
        # 【新增】保存端元演化数据
        save_endmember_data(endmember_evolution, fold_dir)
    
    if len(expert_weights_evolution) > 0:
        # 绘制专家权重演化图
        plot_expert_weights_evolution(
            expert_weights_evolution,
            save_path=f'{fold_dir}/expert_weights_evolution.png'
        )
        
        # 【新增】保存专家权重演化数据
        save_expert_weights_data(expert_weights_evolution, fold_dir)
        
        # ✅ 新增：stats CSV + 3张新图
        print(f"\n{'='*70}")
        print(f"生成解释性分析图表和数据...")
        print(f"{'='*70}")
        
        save_expert_stats_data(expert_weights_evolution, fold_dir)
        
        plot_expert_entropy_evolution(
            expert_weights_evolution,
            save_path=f'{fold_dir}/expert_entropy_evolution.png'
        )
        
        plot_expert_top1_evolution(
            expert_weights_evolution,
            save_path=f'{fold_dir}/expert_top1_evolution.png'
        )
        
        plot_entropy_vs_valdice(
            expert_weights_evolution,
            save_path=f'{fold_dir}/expert_entropy_vs_valdice.png'
        )
    
    print(f"\n{'='*70}")
    print(f"  训练完成！")
    print(f"  Best EMA Dice: {best_ema_dice:.4f}")
    print(f"  模型保存在: {fold_dir}")
    print(f"\n  可视化图表：")
    print(f"    - endmember_evolution.png (端元演化)")
    print(f"    - expert_weights_evolution.png (专家权重演化)")
    print(f"    - expert_entropy_evolution.png (路由熵演化)")
    print(f"    - expert_top1_evolution.png (Top-1专家选择频率)")
    print(f"    - expert_entropy_vs_valdice.png (路由熵 vs 验证Dice对照)")
    print(f"\n  保存的数据文件：")
    print(f"    📊 端元数据：")
    print(f"       - endmember_orthogonality.csv (正交性演化)")
    print(f"       - endmember_spectra_init_final.csv (初始/最终光谱)")
    print(f"       - endmember_correlation_init_final.json (相关性矩阵)")
    print(f"       - endmember_spectra_full.npz (完整演化，可选)")
    print(f"    📊 专家权重数据：")
    print(f"       - expert_weights_evolution.csv (权重演化)")
    print(f"       - expert_weights_evolution.json (权重演化，JSON格式)")
    print(f"    📊 专家统计数据：")
    print(f"       - expert_entropy_evolution.csv (路由熵演化)")
    print(f"       - expert_top1_rate_evolution.csv (Top-1专家频率)")
    print(f"       - expert_weight_std_evolution.csv (权重标准差)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UNet_PPEF++ Training")
    
    # 基础参数
    parser.add_argument("--fold", default=0, type=int, help="交叉验证fold编号")
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size")
    parser.add_argument("--epochs", default=100, type=int, help="训练轮数")
    parser.add_argument("--lr", default=0.00005, type=float, help="学习率")
    parser.add_argument("--lrf", default=0.01, type=float, help="最终学习率系数")
    parser.add_argument("--weight-decay", default=3e-5, type=float, help="权重衰减")
    parser.add_argument("--amp", action='store_true', help="使用混合精度训练")
    parser.add_argument("--use-compile", action='store_true', help="使用 torch.compile() 加速（PyTorch 2.0+）")
    parser.add_argument("--patch-size", default=None, type=int, help="Patch训练大小（None=完整图像）")
    
    # PPEF++特有参数
    parser.add_argument("--num-prototypes", default=4, type=int, help="原型数量")
    parser.add_argument("--dropout-rate", default=0.1, type=float, help="Dropout概率")
    parser.add_argument("--lambda-recon", default=0.1, type=float, help="重建损失权重")
    parser.add_argument("--lambda-smooth", default=0.01, type=float, help="平滑损失权重")
    parser.add_argument("--lambda-div", default=0.01, type=float, help="多样性损失权重")
    parser.add_argument("--lambda-entropy", default=0.001, type=float, help="熵损失权重")
    parser.add_argument("--lambda-orth", default=0.01, type=float, help="原型正交约束权重")  # 【新增】
    
    args = parser.parse_args()
    print(args)
    
    main(args)

