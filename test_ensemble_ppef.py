"""
5折PPEF++模型集成测试脚本（3专家版本）
对每张测试图像，加载5个fold的模型，分别进行滑窗推理，然后对概率图取平均，最后统一阈值化得到最终预测
支持自动检测模型类型（标准UNet或UNet_PPEFPP 3专家版本）
"""

import os
import sys
import torch
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 添加父目录到Python路径，以便导入父目录的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import UNet
from model_ppefpp import UNet_PPEFPP
from dataset import HyperspectralDatasetWithKeys
from utils import calculate_metrics, sliding_window_predict
import scipy.io as sio
from scipy import ndimage as ndi
from scipy.ndimage import binary_closing, binary_fill_holes
from medpy.metric.binary import hd95 as hd95_medpy


def hd95(pred_logits: torch.Tensor, target: torch.Tensor):
    """
    HD95 wrapper using medpy.metric.binary.hd95
    Supports:
        - 2D masks (H, W)
        - 3D masks (D, H, W)
        - batched tensors (B, 1, H, W) or (B, 1, D, H, W)

    Returns:
        mean HD95 over valid samples in batch
    """
    
    # sigmoid / threshold
    if isinstance(pred_logits, torch.Tensor):
        pred = (pred_logits > 0.5).cpu().numpy().astype(np.bool_)
    else:  # already numpy
        pred = (pred_logits > 0.5).astype(np.bool_)

    if isinstance(target, torch.Tensor):
        gt = target.cpu().numpy().astype(np.bool_)
    else:
        gt = target.astype(np.bool_)

    # remove channel dimension if exists
    # shapes it supports:
    #   (B, 1, H, W)
    #   (B, 1, D, H, W)
    #   (H, W)
    #   (D, H, W)

    if pred.ndim == 2:  # (H, W)
        pred = pred[np.newaxis, ...]
        gt = gt[np.newaxis, ...]

    elif pred.ndim == 3 and pred.shape[0] == 1:
        # (1, H, W) -> treat as single sample
        pred = pred
        gt = gt

    elif pred.ndim == 4 and pred.shape[1] == 1:
        # (B, 1, H, W) -> squeeze channel
        pred = pred[:, 0, ...]
        gt = gt[:, 0, ...]

    elif pred.ndim == 5 and pred.shape[1] == 1:
        # (B, 1, D, H, W)
        pred = pred[:, 0, ...]
        gt = gt[:, 0, ...]

    else:
        raise ValueError(f"Unsupported pred shape: {pred.shape}")

    hd_list = []

    for p, g in zip(pred, gt):
        # skip empty masks to avoid medpy crash
        if not p.any() and not g.any():
            continue

        # medpy expects 2D or 3D
        if p.ndim not in (2, 3):
            raise ValueError(f"Each sample must be 2D or 3D, but got {p.shape}")

        v = hd95_medpy(p, g)
        hd_list.append(v)

    if len(hd_list) == 0:
        # no valid samples
        return np.nan

    return float(np.mean(hd_list))


def postprocess_connected_components(pred_mask_np, min_area=20, keep_largest=False):
    """
    连通域去噪 + 可选的最大连通域保留
    
    适用场景：
    - 前景块数量不多（0-3个），小碎片多数是噪点
    - 需要去除小面积的误检区域
    
    Args:
        pred_mask_np: (H, W) 二值预测掩膜，0/1
        min_area: 最小连通域面积阈值（小于此值的连通域将被删除）
        keep_largest: 是否只保留最大的连通域（适用于"一张图只有一个主要病灶"的场景）
    
    Returns:
        cleaned_mask: (H, W) 清洗后的掩膜
    """
    if pred_mask_np.sum() == 0:
        return pred_mask_np
    
    # 连通域标记
    label_im, num = ndi.label(pred_mask_np)
    
    if num == 0:
        return pred_mask_np
    
    # 统计每个连通域的面积
    sizes = ndi.sum(pred_mask_np, label_im, index=range(1, num + 1))
    sizes = np.asarray(sizes)
    
    # 删除小连通域（面积 < min_area）
    remove_ids = np.where(sizes < min_area)[0] + 1
    cleaned = pred_mask_np.copy()
    for rid in remove_ids:
        cleaned[label_im == rid] = 0
    
    # 如果指定只保留最大连通域
    if keep_largest:
        label_im2, num2 = ndi.label(cleaned)
        if num2 == 0:
            return cleaned
        
        sizes2 = ndi.sum(cleaned, label_im2, index=range(1, num2 + 1))
        if len(sizes2) == 0:
            return cleaned
        
        max_id = np.argmax(sizes2) + 1
        largest = np.zeros_like(cleaned)
        largest[label_im2 == max_id] = 1
        return largest.astype(np.uint8)
    
    return cleaned.astype(np.uint8)


def postprocess_morphology(pred_mask_np, radius=1):
    """
    形态学闭运算 + 填洞
    
    作用：
    - 闭运算（binary_closing）：先膨胀再腐蚀，闭合细小间隙和锯齿边缘
    - 填洞（binary_fill_holes）：填充内部空洞
    
    适用场景：
    - 病灶连通但边界锯齿严重
    - 内部有小黑洞需要填补
    
    Args:
        pred_mask_np: (H, W) 二值预测掩膜，0/1
        radius: 结构元素半径（1表示3×3，2表示5×5）
    
    Returns:
        smoothed_mask: (H, W) 平滑后的掩膜
    """
    if pred_mask_np.sum() == 0:
        return pred_mask_np
    
    # 创建结构元素（圆形或方形）
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    
    # 闭运算：膨胀再腐蚀，闭合细小间隙
    closed = binary_closing(pred_mask_np.astype(bool), structure=structure)
    
    # 填洞：填充内部空洞
    filled = binary_fill_holes(closed)
    
    return filled.astype(np.uint8)


def postprocess_ensemble_mask(pred_mask_np, 
                               use_cc=True, min_area=20, keep_largest=False,
                               use_morph=True, morph_radius=1):
    """
    集成后处理流程
    
    推荐流程：
    1. 先连通域去噪（删除小碎片）
    2. 再形态学平滑（闭合边缘、填洞）
    
    Args:
        pred_mask_np: (H, W) 原始预测掩膜
        use_cc: 是否使用连通域去噪
        min_area: 连通域最小面积阈值
        keep_largest: 是否只保留最大连通域
        use_morph: 是否使用形态学平滑
        morph_radius: 形态学结构元素半径
    
    Returns:
        processed_mask: (H, W) 后处理后的掩膜
    """
    processed = pred_mask_np.copy()
    
    # Step 1: 连通域去噪
    if use_cc:
        processed = postprocess_connected_components(
            processed, 
            min_area=min_area, 
            keep_largest=keep_largest
        )
    
    # Step 2: 形态学平滑
    if use_morph:
        processed = postprocess_morphology(
            processed, 
            radius=morph_radius
        )
    
    return processed


def convert_to_serializable(obj):
    """
    递归地将numpy类型转换为Python原生类型，以便JSON序列化
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def load_test_keys(splits_file):
    """加载测试集图像ID"""
    with open(splits_file, 'r') as f:
        data = json.load(f)
    
    # 获取test集（假设在splits中有test字段，或者使用第一个fold的test）
    if 'test' in data:
        test_keys = data['test']
    elif 'splits' in data and len(data['splits']) > 0:
        # 使用第一个fold的test（所有fold的test应该是一样的）
        test_keys = data['splits'][0].get('test', [])
    else:
        raise ValueError("无法从splits文件中找到test集")
    
    return test_keys


def detect_model_type(state_dict):
    """
    检测模型类型（标准UNet或UNet_PPEFPP 3专家版本）
    
    Args:
        state_dict: 模型权重字典
    
    Returns:
        'ppefpp' 或 'standard'
    """
    # 检查是否包含PPEF++模块的特征键（3专家版本）
    ppefpp_indicators = ['unmix_head', 'csse', 'pgacpp', 'spgapp']
    
    for key in state_dict.keys():
        key_lower = key.lower()
        
        # 检测PPEF++（3专家版本）
        for indicator in ppefpp_indicators:
            if indicator in key_lower:
                return 'ppefpp'
    
    return 'standard'


def load_fold_model(fold, base_dir, device, in_channels=40, num_prototypes=4, dropout_rate=0.1):
    """
    加载指定fold的最佳模型（自动检测UNet或UNet_PPEFPP 3专家版本）
    
    Args:
        fold: fold编号 (0-4)
        base_dir: 模型保存的基础目录 (e.g., 'run_results_ppefpp1')
        device: 设备
        in_channels: 输入通道数
        num_prototypes: PPEF++原型数量（每个专家的原型个数，默认4）
        dropout_rate: PPEF++的dropout率
    
    Returns:
        model: 加载好权重并设为eval模式的模型
        model_type: 'ppefpp' 或 'standard'
    """
    # 模型路径
    model_path = os.path.join(base_dir, f'fold_{fold}', 'model_best.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理torch.compile的_orig_mod.前缀
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    
    # 自动检测模型类型
    model_type = detect_model_type(state_dict)
    
    # 创建对应的模型
    if model_type == 'ppefpp':
        print(f"    检测到PPEF++模型（3专家版本）")
        model = UNet_PPEFPP(
            in_channels=in_channels,
            out_channels=1,
            num_prototypes=num_prototypes,
            dropout_rate=dropout_rate,
            use_spgapp=True,
            use_csse=True,
            use_pgacpp=True
        )
    else:
        print(f"    检测到标准UNet模型")
        model = UNet(in_channels=in_channels, out_channels=1)
    
    # 加载权重
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, model_type


def ensemble_inference(models, image, device, patch_size=256, overlap=0.75, num_classes=2):
    """
    对单张图像进行N折模型集成推理
    
    Args:
        models: 模型列表 [model_fold0, model_fold1, ..., model_foldN]
        image: 输入图像 (C, H, W)
        device: 设备
        patch_size: 滑窗patch大小
        overlap: 滑窗重叠比例
        num_classes: 类别数（二分类=2）
    
    Returns:
        prob_ens: 集成后的概率图 (num_classes, H, W)
        pred_mask: 最终预测掩膜 (H, W)
    """
    C, H, W = image.shape
    
    # 初始化集成概率图（二分类：[背景概率, 前景概率]）
    prob_ens = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    
    # 逐个fold推理并累加概率
    for fold_idx, model in enumerate(models):
        # 使用滑窗推理得到单个模型的概率图
        prob_map_fg, _ = sliding_window_predict(
            model, image, device,
            patch_size=patch_size,
            overlap=overlap
        )
        # prob_map_fg: (H, W) - 前景概率
        
        # 构建两类概率（背景 + 前景）
        prob_bg = 1.0 - prob_map_fg  # 背景概率
        
        # 累加到集成概率图
        prob_ens[0] += prob_bg  # 背景类
        prob_ens[1] += prob_map_fg  # 前景类
    
    # 求平均（N个模型的平均概率）
    prob_ens = prob_ens / len(models)
    
    # 从集成概率生成最终预测
    # 方法1：前景概率阈值化（二分类推荐）
    pred_mask = (prob_ens[1] > 0.5).long()  # (H, W)
    
    # 方法2：argmax（多分类）
    # pred_mask = prob_ens.argmax(dim=0)  # (H, W)
    
    return prob_ens, pred_mask


def test_ensemble(args):
    """{args.num_folds}折集成测试主函数"""
    
    # 设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using {device} device for ensemble testing.\n")
    
    # 创建结果保存目录
    ensemble_results_dir = os.path.join(args.base_dir, 'ensemble_test_results')
    os.makedirs(ensemble_results_dir, exist_ok=True)
    predictions_dir = os.path.join(ensemble_results_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    print("="*80)
    print(f"{args.num_folds}折集成测试模式 (patch_size={args.patch_size}×{args.patch_size})")
    print(f"模型基础目录: {args.base_dir}")
    print(f"结果保存目录: {ensemble_results_dir}")
    print("="*80)
    print(f"\n后处理配置:")
    print(f"  连通域去噪: {'✓ 启用' if args.use_cc else '✗ 禁用'}")
    if args.use_cc:
        print(f"    - 最小面积阈值: {args.min_area} 像素")
        print(f"    - 只保留最大连通域: {'是' if args.keep_largest else '否'}")
    print(f"  形态学平滑: {'✓ 启用' if args.use_morph else '✗ 禁用'}")
    if args.use_morph:
        print(f"    - 结构元素半径: {args.morph_radius} (核大小: {2*args.morph_radius+1}×{2*args.morph_radius+1})")
    print()
    
    # 加载N个fold的模型
    print(f"正在加载{args.num_folds}个fold的模型...")
    models = []
    model_types = []
    for fold in range(args.num_folds):
        print(f"  加载 Fold {fold} 模型...")
        try:
            model, model_type = load_fold_model(
                fold=fold,
                base_dir=args.base_dir,
                device=device,
                in_channels=args.in_channels,
                num_prototypes=args.num_prototypes,
                dropout_rate=args.dropout_rate
            )
            models.append(model)
            model_types.append(model_type)
            print(f"  ✓ Fold {fold} 模型加载成功 (类型: {model_type.upper()})")
        except Exception as e:
            print(f"  ✗ Fold {fold} 模型加载失败: {e}")
            sys.exit(1)
    
    # 检查模型类型一致性
    if len(set(model_types)) > 1:
        print(f"\n⚠️  警告：检测到不同类型的模型！")
        for i, mt in enumerate(model_types):
            print(f"    Fold {i}: {mt.upper()}")
        print()
    
    print(f"\n✓ 成功加载 {len(models)} 个模型 (类型: {model_types[0].upper()})\n")
    
    # 加载测试集
    print("正在加载测试集...")
    test_keys = load_test_keys(args.splits_file)
    print(f"✓ 测试集: {len(test_keys)} 张图像\n")
    
    # 创建测试集DataLoader
    testDataset = HyperspectralDatasetWithKeys(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        keys=test_keys,
        transform=None
    )
    
    testLoader = DataLoader(
        testDataset,
        batch_size=1,  # 集成测试时每次处理一张图
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"开始集成测试（逐案例评估）...\n")
    
    # 逐案例测试
    num_classes = 2
    metric_per_case = []
    
    with torch.no_grad():
        for idx, (test_image, test_target, case_name_list) in enumerate(testLoader):
            case_name = case_name_list[0]
            
            print(f"[{idx+1}/{len(testLoader)}] 测试: {case_name} ...", end=' ')
            
            # 获取图像和真值
            img = test_image[0]  # (C, H, W)
            gt = test_target[0].to(device)  # (H, W)
            
            # N折模型集成推理
            prob_ens, pred_mask = ensemble_inference(
                models=models,
                image=img,
                device=device,
                patch_size=args.patch_size,
                overlap=0.75,
                num_classes=num_classes
            )
            
            # 转为numpy用于指标计算
            pred_mask_np = pred_mask.cpu().numpy()
            gt_np = gt.cpu().numpy()
            
            # ========== 后处理（连通域去噪 + 形态学平滑）==========
            pred_mask_np = postprocess_ensemble_mask(
                pred_mask_np,
                use_cc=args.use_cc,
                min_area=args.min_area,
                keep_largest=args.keep_largest,
                use_morph=args.use_morph,
                morph_radius=args.morph_radius
            )
            
            # 计算指标
            case_metrics = calculate_metrics(
                torch.from_numpy(pred_mask_np).unsqueeze(0),
                gt.unsqueeze(0).cpu(),
                num_classes
            )
            
            # 计算 HD95（基于后处理后的预测）
            try:
                hd95_val = hd95(pred_mask_np, gt_np)
            except:
                hd95_val = np.inf  # 如果计算失败（如空掩膜），设为无穷大
            
            # 保存案例结果
            case_result = {
                'case_name': case_name,
                'dice': case_metrics['dice'],
                'iou': case_metrics['iou'],
                'accuracy': case_metrics['accuracy'],
                'hd95': float(hd95_val) if not np.isinf(hd95_val) else np.inf,
                'metrics': case_metrics
            }
            metric_per_case.append(case_result)
            
            print(f"Dice: {case_metrics['dice']:.4f}, IoU: {case_metrics['iou']:.4f}, Acc: {case_metrics['accuracy']:.4f}, HD95: {hd95_val:.2f}" if not np.isinf(hd95_val) else f"Dice: {case_metrics['dice']:.4f}, IoU: {case_metrics['iou']:.4f}, Acc: {case_metrics['accuracy']:.4f}, HD95: inf")
            
            # 保存预测图像（不加文字描述）
            plt.figure(figsize=(pred_mask_np.shape[1]/100, pred_mask_np.shape[0]/100), dpi=100)
            plt.imshow(pred_mask_np, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(os.path.join(predictions_dir, f'{case_name}_ensemble_pred.png'), 
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 可选：保存详细对比图
            if args.save_predictions:
                plt.figure(figsize=(18, 6))
                
                # 预测结果
                plt.subplot(1, 4, 1)
                plt.imshow(pred_mask_np, cmap='gray', vmin=0, vmax=1)
                plt.title(f'Ensemble Prediction\nDice: {case_metrics["dice"]:.4f}', fontsize=12)
                plt.axis('off')
                
                # 真值
                plt.subplot(1, 4, 2)
                plt.imshow(gt_np, cmap='gray', vmin=0, vmax=1)
                plt.title('Ground Truth', fontsize=12)
                plt.axis('off')
                
                # 叠加图
                plt.subplot(1, 4, 3)
                overlay = np.zeros((*pred_mask_np.shape, 3))
                overlay[pred_mask_np == 1] = [1, 0, 0]  # 红色：预测
                overlay[gt_np == 1] += [0, 1, 0]  # 绿色：真值
                plt.imshow(overlay)
                plt.title('Overlay\n(Red=Pred, Green=GT, Yellow=Both)', fontsize=12)
                plt.axis('off')
                
                # 错误图
                plt.subplot(1, 4, 4)
                error_map = np.zeros((*pred_mask_np.shape, 3))
                tp = (pred_mask_np == 1) & (gt_np == 1)
                fp = (pred_mask_np == 1) & (gt_np == 0)
                fn = (pred_mask_np == 0) & (gt_np == 1)
                tn = (pred_mask_np == 0) & (gt_np == 0)
                error_map[tp] = [1, 1, 1]  # 白色：TP
                error_map[fp] = [1, 0, 0]  # 红色：FP
                error_map[fn] = [0, 0, 1]  # 蓝色：FN
                error_map[tn] = [0, 0, 0]  # 黑色：TN
                plt.imshow(error_map)
                plt.title('Error Map\n(Red=FP, Blue=FN, White=TP)', fontsize=12)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(predictions_dir, f'{case_name}_ensemble_detailed.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"\n{'='*80}")
    print(f"{args.num_folds}折集成测试完成！共测试 {len(metric_per_case)} 个案例")
    print(f"{'='*80}\n")
    
    # 计算汇总指标（均值和标准差）
    foreground_metrics = {}
    foreground_metrics_std = {}
    background_metrics = {}
    background_metrics_std = {}
    class_metric_names = ['dice', 'iou', 'precision', 'recall', 'f1', 'sensitivity', 'specificity']
    
    for metric_name in class_metric_names:
        fg_values = []
        bg_values = []
        
        for case in metric_per_case:
            class_metrics = case['metrics']['class_metrics']
            
            # 查找前景类
            for key in class_metrics:
                if '1' in str(key):
                    val = class_metrics[key][metric_name]
                    if not np.isnan(val):
                        fg_values.append(val)
                    break
            
            # 查找背景类
            for key in class_metrics:
                if '0' in str(key) and '1' not in str(key):
                    val = class_metrics[key][metric_name]
                    if not np.isnan(val):
                        bg_values.append(val)
                    break
        
        # 计算均值和标准差
        foreground_metrics[metric_name] = float(np.mean(fg_values)) if len(fg_values) > 0 else np.nan
        foreground_metrics_std[metric_name] = float(np.std(fg_values, ddof=1)) if len(fg_values) > 1 else 0.0
        background_metrics[metric_name] = float(np.mean(bg_values)) if len(bg_values) > 0 else np.nan
        background_metrics_std[metric_name] = float(np.std(bg_values, ddof=1)) if len(bg_values) > 1 else 0.0
    
    # 计算前景类HD95（过滤掉inf值）
    fg_hd95_values = [case['hd95'] for case in metric_per_case]
    valid_fg_hd95 = [h for h in fg_hd95_values if not np.isinf(h)]
    foreground_metrics['hd95'] = float(np.mean(valid_fg_hd95)) if len(valid_fg_hd95) > 0 else np.inf
    foreground_metrics_std['hd95'] = float(np.std(valid_fg_hd95, ddof=1)) if len(valid_fg_hd95) > 1 else 0.0
    
    # 计算整体指标（均值和标准差）
    accuracy_values = [case['metrics']['accuracy'] for case in metric_per_case]
    overall_accuracy = float(np.mean(accuracy_values)) if len(accuracy_values) > 0 else np.nan
    overall_accuracy_std = float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else 0.0
    
    dice_values = [case['dice'] for case in metric_per_case]
    overall_dice = float(np.mean(dice_values))
    overall_dice_std = float(np.std(dice_values, ddof=1)) if len(dice_values) > 1 else 0.0
    
    iou_values = [case['iou'] for case in metric_per_case]
    overall_iou = float(np.mean(iou_values))
    overall_iou_std = float(np.std(iou_values, ddof=1)) if len(iou_values) > 1 else 0.0
    
    # 计算HD95平均值和标准差（过滤掉inf值）
    hd95_values = [case['hd95'] for case in metric_per_case]
    valid_hd95 = [h for h in hd95_values if not np.isinf(h)]
    mean_hd95 = float(np.mean(valid_hd95)) if len(valid_hd95) > 0 else np.inf
    std_hd95 = float(np.std(valid_hd95, ddof=1)) if len(valid_hd95) > 1 else 0.0
    
    overall_metrics = {
        'accuracy': overall_accuracy,
        'accuracy_std': overall_accuracy_std,
        'dice': overall_dice,
        'dice_std': overall_dice_std,
        'iou': overall_iou,
        'iou_std': overall_iou_std,
        'hd95': mean_hd95,
        'hd95_std': std_hd95
    }
    
    # 打印结果（均值±标准差格式）
    print("="*80)
    print(f"{args.num_folds}折集成测试结果汇总")
    print("="*80)
    print(f"\n整体指标:")
    print(f"  Accuracy: {overall_metrics['accuracy']:.4f} ± {overall_metrics['accuracy_std']:.4f}")
    print(f"  Dice:     {overall_metrics['dice']:.4f} ± {overall_metrics['dice_std']:.4f}")
    print(f"  IoU:      {overall_metrics['iou']:.4f} ± {overall_metrics['iou_std']:.4f}")
    if not np.isinf(mean_hd95):
        print(f"  HD95:     {mean_hd95:.4f} ± {std_hd95:.4f} ({len(valid_hd95)}/{len(hd95_values)} valid)")
    else:
        print(f"  HD95:     inf (no valid values)")
    
    print(f"\n前景类指标:")
    for metric_name in class_metric_names:
        val = foreground_metrics.get(metric_name, np.nan)
        std_val = foreground_metrics_std.get(metric_name, 0.0)
        print(f"  {metric_name.capitalize():12s}: {val:.4f} ± {std_val:.4f}")
    # 添加HD95输出
    fg_hd95 = foreground_metrics.get('hd95', np.inf)
    fg_hd95_std = foreground_metrics_std.get('hd95', 0.0)
    if not np.isinf(fg_hd95):
        print(f"  {'Hd95':12s}: {fg_hd95:.4f} ± {fg_hd95_std:.4f}")
    else:
        print(f"  {'Hd95':12s}: inf")
    
    print(f"\n背景类指标:")
    for metric_name in class_metric_names:
        val = background_metrics.get(metric_name, np.nan)
        std_val = background_metrics_std.get(metric_name, 0.0)
        print(f"  {metric_name.capitalize():12s}: {val:.4f} ± {std_val:.4f}")
    
    # 保存JSON结果
    results = {
        'ensemble_info': {
            'num_folds': args.num_folds,
            'model_type': model_types[0],
            'patch_size': args.patch_size,
            'overlap': 0.75,
            'base_dir': args.base_dir,
            'num_prototypes': args.num_prototypes if model_types[0] == 'ppefpp' else None,
            'dropout_rate': args.dropout_rate if model_types[0] == 'ppefpp' else None
        },
        'postprocessing': {
            'use_cc': args.use_cc,
            'min_area': args.min_area if args.use_cc else None,
            'keep_largest': args.keep_largest if args.use_cc else None,
            'use_morph': args.use_morph,
            'morph_radius': args.morph_radius if args.use_morph else None
        },
        'overall_metrics': overall_metrics,
        'foreground_metrics': foreground_metrics,
        'foreground_metrics_std': foreground_metrics_std,
        'background_metrics': background_metrics,
        'background_metrics_std': background_metrics_std,
        'per_case_results': metric_per_case
    }
    
    # 转换为可序列化的格式（处理numpy类型）
    results = convert_to_serializable(results)
    
    json_path = os.path.join(ensemble_results_dir, 'ensemble_test_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON结果已保存: {json_path}")
    
    # 保存文本结果
    txt_path = os.path.join(ensemble_results_dir, 'ensemble_test_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{args.num_folds}折集成测试结果\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"模型类型: {model_types[0].upper()}\n")
        if model_types[0] == 'ppefpp':
            f.write(f"  - 专家数量: 3 (Spectral + Spatial + Region)\n")
            f.write(f"  - 原型数量: {args.num_prototypes} (每个专家的原型个数)\n")
            f.write(f"  - Dropout率: {args.dropout_rate}\n")
            f.write(f"  - 特性: CSSE (3专家) + PGAC++ + SPGA++ + Unmixing Head\n")
        f.write("\n")
        
        f.write(f"后处理配置:\n")
        f.write(f"  连通域去噪: {'启用' if args.use_cc else '禁用'}\n")
        if args.use_cc:
            f.write(f"    - 最小面积阈值: {args.min_area} 像素\n")
            f.write(f"    - 只保留最大连通域: {'是' if args.keep_largest else '否'}\n")
        f.write(f"  形态学平滑: {'启用' if args.use_morph else '禁用'}\n")
        if args.use_morph:
            f.write(f"    - 结构元素半径: {args.morph_radius} (核大小: {2*args.morph_radius+1}×{2*args.morph_radius+1})\n")
        f.write("\n")
        
        f.write(f"整体指标:\n")
        f.write(f"  Accuracy: {overall_metrics['accuracy']:.4f} ± {overall_metrics['accuracy_std']:.4f}\n")
        f.write(f"  Dice:     {overall_metrics['dice']:.4f} ± {overall_metrics['dice_std']:.4f}\n")
        f.write(f"  IoU:      {overall_metrics['iou']:.4f} ± {overall_metrics['iou_std']:.4f}\n")
        if not np.isinf(mean_hd95):
            f.write(f"  HD95:     {mean_hd95:.4f} ± {std_hd95:.4f} ({len(valid_hd95)}/{len(hd95_values)} valid)\n\n")
        else:
            f.write(f"  HD95:     inf (no valid values)\n\n")
        
        f.write(f"前景类指标:\n")
        for metric_name in class_metric_names:
            val = foreground_metrics.get(metric_name, np.nan)
            std_val = foreground_metrics_std.get(metric_name, 0.0)
            f.write(f"  {metric_name.capitalize():12s}: {val:.4f} ± {std_val:.4f}\n")
        # 添加HD95输出
        fg_hd95 = foreground_metrics.get('hd95', np.inf)
        fg_hd95_std = foreground_metrics_std.get('hd95', 0.0)
        if not np.isinf(fg_hd95):
            f.write(f"  {'Hd95':12s}: {fg_hd95:.4f} ± {fg_hd95_std:.4f}\n")
        else:
            f.write(f"  {'Hd95':12s}: inf\n")
        
        f.write(f"\n背景类指标:\n")
        for metric_name in class_metric_names:
            val = background_metrics.get(metric_name, np.nan)
            std_val = background_metrics_std.get(metric_name, 0.0)
            f.write(f"  {metric_name.capitalize():12s}: {val:.4f} ± {std_val:.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"逐案例结果:\n")
        f.write(f"{'='*80}\n\n")
        
        for i, case in enumerate(metric_per_case, 1):
            f.write(f"[{i}/{len(metric_per_case)}] {case['case_name']}\n")
            f.write(f"  Dice:     {case['dice']:.4f}\n")
            f.write(f"  IoU:      {case['iou']:.4f}\n")
            f.write(f"  Accuracy: {case['accuracy']:.4f}\n")
            hd95_case = case['hd95']
            if not np.isinf(hd95_case):
                f.write(f"  HD95:     {hd95_case:.4f}\n\n")
            else:
                f.write(f"  HD95:     inf\n\n")
    
    print(f"✓ 文本结果已保存: {txt_path}")
    print(f"✓ 预测图像已保存到: {predictions_dir}\n")
    
    print("="*80)
    print("集成测试完成！")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N折模型集成测试（支持UNet和PPEF++ 3专家版本）')
    
    # 必需参数
    parser.add_argument('--num-folds', type=int, default=5,
                        help='fold数量（默认5，根据训练的fold数量设置）')
    parser.add_argument('--base-dir', type=str, default='run_results_ppefpp22',
                        help='模型保存的基础目录（包含fold_0, fold_1等子目录）')
    parser.add_argument('--splits-file', type=str, default='/data/CXY/gwj/WUnet/2DIM/splits_im_patch.json',
                        help='数据划分文件路径')
    parser.add_argument('--image-dir', type=str, 
                        default='/home/ubuntu/dataset_Med/PLGC/IM/IM_HSI_mat',
                        help='图像数据目录')
    parser.add_argument('--mask-dir', type=str,
                        default='/home/ubuntu/dataset_Med/PLGC/IM/IM_label_mat',
                        help='标签数据目录')
    
    # 模型参数
    parser.add_argument('--in-channels', type=int, default=40,
                        help='输入通道数（高光谱波段数）')
    parser.add_argument('--patch-size', type=int, default=256,
                        help='滑窗patch大小')
    
    # PPEF++参数（3专家版本）
    parser.add_argument('--num-prototypes', type=int, default=4,
                        help='PPEF++原型数量（每个专家的原型个数，默认4）')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='PPEF++ Dropout率')
    
    # 后处理参数
    parser.add_argument('--use-cc', action='store_true', default=True,
                        help='是否使用连通域去噪（默认开启）')
    parser.add_argument('--no-cc', dest='use_cc', action='store_false',
                        help='禁用连通域去噪')
    parser.add_argument('--min-area', type=int, default=20,
                        help='连通域最小面积阈值（小于此值的连通域将被删除，默认20）')
    parser.add_argument('--keep-largest', action='store_true',
                        help='是否只保留最大连通域（适用于单病灶场景）')
    parser.add_argument('--use-morph', action='store_true', default=True,
                        help='是否使用形态学平滑（闭运算+填洞，默认开启）')
    parser.add_argument('--no-morph', dest='use_morph', action='store_false',
                        help='禁用形态学平滑')
    parser.add_argument('--morph-radius', type=int, default=1,
                        help='形态学结构元素半径（1表示3×3，2表示5×5，默认1）')
    
    # 其他参数
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID（-1表示使用CPU）')
    parser.add_argument('--save-predictions', action='store_true',
                        help='是否保存详细的预测对比图')
    
    args = parser.parse_args()
    
    # 运行集成测试
    test_ensemble(args)

