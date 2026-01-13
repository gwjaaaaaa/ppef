import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Loss函数实现（参考improved_professional_difabal.py和111/loss.py）
# ============================================================================

def dice_loss_fn(outputs, targets):
    """
    Dice损失函数（完全按照111/loss.py实现）
    
    Args:
        outputs: 模型输出logits，shape: (B, 1, H, W)
        targets: 目标mask，shape: (B, 1, H, W)
    
    Returns:
        dice_loss: 1 - dice_coefficient
    """
    # 将outputs转换为概率（因为BCEWithLogitsLoss没有过sigmoid，所以要加）
    inputs = torch.sigmoid(outputs)
    
    # 平滑项（避免除零）
    smooth = 1e-5
    
    # Flatten成(B, H*W) - 与111/loss.py完全一致
    inputs_flat = inputs.view(inputs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    
    # 计算Dice系数 - 与111/loss.py完全一致
    intersection = (inputs_flat * targets_flat).sum(1)
    dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum(1) + targets_flat.sum(1) + smooth)
    dice_loss = dice_loss.mean()
    
    return dice_loss


# 保留BCEDiceLoss类用于兼容性（但推荐使用分离的方式）
class BCEDiceLoss(nn.Module):
    """
    BCE + Dice 组合损失函数（完全按照111/loss.py实现）
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        """
        Args:
            bce_weight: BCE损失权重（默认0.5，与111一致）
            dice_weight: Dice损失权重（默认0.5，与111一致）
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出（logits），shape: (B, 1, H, W)
            targets: 目标mask，shape: (B, 1, H, W)，值为0或1
        Returns:
            loss: 组合损失（0.5*BCE + 0.5*Dice，与111一致）
        """
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss - 与111/loss.py完全一致
        dice_loss = dice_loss_fn(inputs, targets)
        
        # 总Loss - 权重0.5/0.5与111一致
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss


# ============================================================================
# 指标计算函数（参考improved_professional_difabal.py）
# ============================================================================

def dice_coefficient(pred, target, threshold=0.5):
    """
    计算 Dice 系数（完全按照111/metrics.py实现）
    用于验证阶段计算真实的Dice分数
    
    Args:
        pred: 模型输出logits，shape: (B, 1, H, W)
        target: 目标mask，shape: (B, 1, H, W)
        threshold: 二值化阈值，默认0.5
    
    Returns:
        dice: Dice系数，范围0-1
    """
    pred = torch.sigmoid(pred)  # 先做sigmoid，因为输出是logits
    pred = (pred > threshold).float()

    smooth = 1e-5  # 防止除0
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def hd95_numpy(pred_mask, true_mask):
    """
    计算 95% Hausdorff Distance (HD95) - 优化版本
    
    HD95 是表面距离的 95th percentile，对离群点更鲁棒。
    它衡量预测边界和真实边界之间的距离。
    
    使用向量化计算和采样策略来加速大图像的计算。
    
    Args:
        pred_mask: 预测掩码 (H, W), 值为0或1
        true_mask: 真实掩码 (H, W), 值为0或1
    
    Returns:
        hd95: 95% Hausdorff距离 (像素单位)
              返回 np.inf 如果其中一个mask为空
    """
    # 确保输入是numpy数组
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(true_mask):
        true_mask = true_mask.cpu().numpy()
    
    pred_mask = pred_mask.astype(np.uint8)
    true_mask = true_mask.astype(np.uint8)
    
    # 提取前景点
    pred_points = np.argwhere(pred_mask > 0)
    true_points = np.argwhere(true_mask > 0)
    
    # 如果任一mask为空，返回无穷大
    if len(pred_points) == 0 or len(true_points) == 0:
        return np.inf
    
    # 优化策略：如果点太多（>10000），进行采样以加速
    max_points = 10000
    if len(pred_points) > max_points:
        indices = np.random.choice(len(pred_points), max_points, replace=False)
        pred_points = pred_points[indices]
    if len(true_points) > max_points:
        indices = np.random.choice(len(true_points), max_points, replace=False)
        true_points = true_points[indices]
    
    # 使用 scipy 的 cdist 进行高效的距离计算（向量化）
    from scipy.spatial.distance import cdist
    
    # 计算距离矩阵（分批处理以节省内存）
    batch_size = 1000
    distances_pred_to_true = []
    
    for i in range(0, len(pred_points), batch_size):
        batch = pred_points[i:i+batch_size]
        # 计算batch中每个点到所有true_points的最小距离
        dists = cdist(batch, true_points, metric='euclidean')
        min_dists = np.min(dists, axis=1)
        distances_pred_to_true.extend(min_dists)
    
    distances_true_to_pred = []
    for i in range(0, len(true_points), batch_size):
        batch = true_points[i:i+batch_size]
        # 计算batch中每个点到所有pred_points的最小距离
        dists = cdist(batch, pred_points, metric='euclidean')
        min_dists = np.min(dists, axis=1)
        distances_true_to_pred.extend(min_dists)
    
    # 合并所有距离
    all_distances = np.concatenate([distances_pred_to_true, distances_true_to_pred])
    
    # 计算95th percentile
    hd95 = np.percentile(all_distances, 95)
    
    return hd95


def calculate_comprehensive_metrics(pred_mask, true_mask, num_classes=2):
    """
    计算全面的分割指标（参考improved_professional_difabal.py）
    
    Args:
        pred_mask: 预测掩码 (H, W) 或 (B, H, W)
        true_mask: 真实掩码 (H, W) 或 (B, H, W)
        num_classes: 类别数
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 转换为numpy数组
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(true_mask):
        true_mask = true_mask.cpu().numpy()
    
    # 确保尺寸匹配
    if pred_mask.shape != true_mask.shape:
        print(f"    警告：尺寸不匹配 {pred_mask.shape} vs {true_mask.shape}")
        return {}
    
    # 展平
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # 基本指标
    correct = (pred_flat == true_flat)
    pixel_accuracy = np.mean(correct)
    accuracy = pixel_accuracy  # 统一命名
    
    # 获取类别
    classes = np.unique(np.concatenate([pred_flat, true_flat]))
    
    # 每类指标
    class_metrics = {}
    ious = []
    dices = []
    precisions = []
    recalls = []
    f1_scores = []
    sensitivities = []
    specificities = []
    hd95_list = []  # HD95列表
    
    for cls in classes:
        # 二值化
        pred_binary = (pred_flat == cls)
        true_binary = (true_flat == cls)
        
        # TP, FP, FN, TN
        tp = np.sum(pred_binary & true_binary)
        fp = np.sum(pred_binary & ~true_binary)
        fn = np.sum(~pred_binary & true_binary)
        tn = np.sum(~pred_binary & ~true_binary)
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-8)
        ious.append(iou)
        
        # Dice
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        dices.append(dice)
        
        # Precision & Recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
        
        # Sensitivity (Recall) & Specificity
        sensitivity = tp / (tp + fn + 1e-8)  # 敏感性
        specificity = tn / (tn + fp + 1e-8)  # 特异性
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        
        # HD95 (Hausdorff Distance 95%)
        # 重建二值mask（从flat恢复shape）
        pred_binary_mask = pred_mask == cls  # 使用原始shape
        true_binary_mask = true_mask == cls
        hd95_val = hd95_numpy(pred_binary_mask, true_binary_mask)
        hd95_list.append(hd95_val)
        
        class_metrics[f'class_{cls}'] = {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'hd95': hd95_val,  # 添加HD95
            'support': np.sum(true_binary)
        }
    
    # 宏平均
    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    
    # HD95 平均值（过滤掉 inf）
    valid_hd95 = [h for h in hd95_list if not np.isinf(h)]
    mean_hd95 = np.mean(valid_hd95) if valid_hd95 else np.inf
    
    # 对于医学图像分割，我们关心前景类(class_1)的敏感性、特异性和HD95
    if 1 in classes:
        # 使用前景类的指标
        foreground_idx = list(classes).index(1)
        mean_sensitivity = sensitivities[foreground_idx]
        mean_specificity = specificities[foreground_idx]
        foreground_hd95 = hd95_list[foreground_idx]
    else:
        # 如果没有class_1，使用最后一个类别（通常是前景）
        mean_sensitivity = sensitivities[-1] if sensitivities else 0
        mean_specificity = specificities[-1] if specificities else 0
        foreground_hd95 = hd95_list[-1] if hd95_list else np.inf
    
    # 数据利用率 (Data Utilization Rate)
    total_pixels = len(pred_flat)
    labeled_pixels = np.sum(true_flat > 0)  # 非背景像素
    data_utilization_rate = labeled_pixels / total_pixels
    
    # AA (Average Accuracy) - 各类别召回率的平均
    class_recalls = []
    for cls in classes:
        pred_binary = (pred_flat == cls)
        true_binary = (true_flat == cls)
        tp = np.sum(pred_binary & true_binary)
        fn = np.sum(~pred_binary & true_binary)
        recall = tp / (tp + fn + 1e-8)  # 类别召回率
        class_recalls.append(recall)
    aa = np.mean(class_recalls)
    
    # 混淆矩阵
    cm = confusion_matrix(true_flat, pred_flat, labels=classes)
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'accuracy': accuracy,  # 统一命名
        'iou': mean_iou,
        'dice': mean_dice,
        'precision': mean_precision,
        'recall': mean_recall,
        'f1': mean_f1,
        'sensitivity': mean_sensitivity,
        'specificity': mean_specificity,
        'hd95': foreground_hd95,  # 前景类的HD95（或最后一个类别）
        'mean_hd95': mean_hd95,   # 所有类别的平均HD95
        'aa': aa,
        'data_utilization_rate': data_utilization_rate,
        'class_metrics': class_metrics,
        'confusion_matrix': cm.tolist(),
        'classes': classes.tolist()
    }


# 兼容旧名称
calculate_metrics = calculate_comprehensive_metrics


# ============================================================================
# EMA类（指数移动平均）
# ============================================================================

class EMA:
    """指数移动平均（Exponential Moving Average）"""
    def __init__(self, decay=0.9):
        """
        Args:
            decay: 衰减系数，通常取0.9-0.999
        """
        self.decay = decay
        self.shadow = {}
    
    def update(self, name, value):
        """
        更新EMA值
        
        Args:
            name: 指标名称
            value: 当前值
        """
        if name not in self.shadow:
            self.shadow[name] = value
        else:
            self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * value
        return self.shadow[name]
    
    def get(self, name):
        """获取EMA值"""
        return self.shadow.get(name, 0.0)
    
    def reset(self):
        """重置所有EMA值"""
        self.shadow = {}


# 训练一个 epoch
def train_one_epoch(model, optim, train_loader, test_loader, device, scaler=None):
    """
    训练一个epoch，支持混合精度训练
    
    Args:
        model: 模型
        optim: 优化器
        train_loader: 训练集DataLoader（patch-wise）
        test_loader: 测试集DataLoader（未使用）
        device: 设备
        scaler: GradScaler for mixed precision training (可选)
    
    Returns:
        train_loss, test_loss, lr
    """
    from torch.cuda.amp import autocast
    
    # 组合损失函数（参考improved_professional_difabal.py）
    # 分别定义BCE和Dice，在训练循环中分别计算后组合
    bce_loss = nn.BCEWithLogitsLoss()
    # dice_loss使用dice_loss_fn函数
    
    use_amp = scaler is not None  # 是否使用混合精度

    model.train()
    train_running_loss = 0.0        # 训练集的总损失
    train_num = 0                   # 训练样本总数
    
    # 移除tqdm进度条，提高GPU利用率
    for train_image, train_target, _ in train_loader:  # HyperspectralDataset 返回3个值
        # 单通道二分类：target需要unsqueeze(1)变成 (B, 1, H, W)
        train_image = train_image.to(device, non_blocking=True)
        train_target = train_target.to(device, non_blocking=True).unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        optim.zero_grad(set_to_none=True)  # 梯度清零（set_to_none=True更高效）
        
        # 混合精度训练
        if use_amp:
            with autocast():  # 自动混合精度
                train_output = model(train_image)               # 前向传播，输出 (B, 1, H, W)
                
                # 分别计算BCE和Dice损失（与111/loss.py一致，权重0.5/0.5）
                bce_loss_val = bce_loss(train_output, train_target.float())
                dice_loss_val = dice_loss_fn(train_output, train_target.float())
                loss = 0.5 * bce_loss_val + 0.5 * dice_loss_val  # 组合损失，权重与111一致
            
            # 使用scaler进行梯度缩放
            scaler.scale(loss).backward()                       # 反向传播（缩放后的loss）
            scaler.unscale_(optim)                              # 取消缩放以进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)  # 梯度裁剪
            scaler.step(optim)                                  # 更新参数
            scaler.update()                                     # 更新scaler
        else:
            # 标准FP32训练
            train_output = model(train_image)               # 前向传播，输出 (B, 1, H, W)
            
            # 分别计算BCE和Dice损失（与111/loss.py一致，权重0.5/0.5）
            bce_loss_val = bce_loss(train_output, train_target.float())
            dice_loss_val = dice_loss_fn(train_output, train_target.float())
            loss = 0.5 * bce_loss_val + 0.5 * dice_loss_val  # 组合损失，权重与111一致
            
            loss.backward()                                 # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)  # 梯度裁剪（参考nnUNet）
            optim.step()                                    # 梯度更新

        # 累加loss：loss.item()是batch平均loss，需要乘以batch_size还原为总loss
        batch_size = train_image.size(0)
        train_running_loss += loss.item() * batch_size
        train_num += batch_size

    lr = optim.param_groups[0]["lr"]
    # 训练阶段不进行验证，返回0作为test_loss占位
    return train_running_loss/train_num, 0.0, lr


# 滑窗预测函数（用于整图推理）- 使用重叠滑窗+高斯权重融合
def sliding_window_predict(model, image, device, patch_size=256, overlap=0.5):
    """
    使用重叠滑窗对整图进行预测，采用高斯权重融合消除拼接痕迹
    
    Args:
        model: 模型
        image: 输入图像tensor, shape (C, H, W)
        device: 设备
        patch_size: patch尺寸
        overlap: 重叠比例（0.5表示50%重叠，stride=patch_size*0.5）
    
    Returns:
        prob_map: 概率图, shape (H, W)
        pred_mask: 预测掩膜, shape (H, W)
    """
    model.eval()
    C, H, W = image.shape
    ps = patch_size
    
    # 计算stride（重叠滑窗）
    stride = int(ps * (1 - overlap))
    if stride <= 0:
        stride = ps  # 防守式
    
    # 初始化概率累加图和权重累加图
    prob_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    weight_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    # 创建高斯权重窗（中心权重大，边缘权重小）
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, ps, device=device),
        torch.linspace(-1, 1, ps, device=device),
        indexing='ij'
    )
    dist = torch.sqrt(xx**2 + yy**2)
    gaussian_weight = torch.exp(- (dist**2) / 0.5)  # 中心权重大，边缘权重小
    
    # 将图像移到设备上
    image = image.to(device)
    
    with torch.no_grad():
        # 重叠滑窗遍历（边界时回拉起点，确保永远提取完整的ps×ps patch）
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # 关键：到边界时将起点"回拉"，确保提取的patch是完整的ps×ps
                y0 = min(y, H - ps)  # 如果y+ps>H，则y0=H-ps（回拉）
                x0 = min(x, W - ps)  # 如果x+ps>W，则x0=W-ps（回拉）
                
                # 提取完整的ps×ps patch（永远不需要padding）
                patch = image[:, y0:y0+ps, x0:x0+ps]  # (C, ps, ps)
                
                # 前向传播
                patch = patch.unsqueeze(0)  # (1, C, ps, ps)
                
                # 兼容PPEF++模型和普通UNet模型
                try:
                    # PPEF++模型：显式传入return_unmixing=False
                    out = model(patch, return_unmixing=False)  # (1, 1, ps, ps)
                except TypeError:
                    # 普通UNet模型：不支持return_unmixing参数
                    out = model(patch)  # (1, 1, ps, ps)
                
                prob = torch.sigmoid(out)[0, 0]  # (ps, ps)
                
                # 加权累加到概率图（使用完整的高斯权重）
                prob_map[y0:y0+ps, x0:x0+ps] += prob * gaussian_weight
                weight_map[y0:y0+ps, x0:x0+ps] += gaussian_weight
        
        # 归一化概率图（加权平均）
        prob_map = prob_map / torch.clamp(weight_map, min=1e-6)
        
        # 最后统一阈值化
        pred_mask = (prob_map > 0.5).long()  # (H, W)
    
    return prob_map, pred_mask


# 计算验证损失和Dice（完整图像直接前向传播）
def calculate_val_loss_and_dice(model, val_loader, device, scaler=None):
    """
    计算验证集上的损失和真实Dice（完全按照111/train.py实现）
    
    与训练不同：
    - 训练：使用256×256 patch，每个batch包含2个patch
    - 验证：使用完整1024×1280图像，每个batch包含2张完整图像
    - UNet是全卷积网络，可以接受任意尺寸输入
    
    Args:
        model: 模型
        val_loader: 验证集DataLoader（完整图像）
        device: 设备
        scaler: GradScaler（可选，用于混合精度验证）
    
    Returns:
        val_loss: float, 验证损失（样本平均，0-1范围）
        val_dice: float, 真实Dice系数（0-1范围，应该>0.5）
    """
    from torch.cuda.amp import autocast
    
    # 使用BCEDiceLoss，与111/train.py一致
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    
    use_amp = scaler is not None
    
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        for val_inputs, val_targets, _ in val_loader:
            # val_inputs: (B, C, H, W) - 完整图像
            # val_targets: (B, H, W) - 完整标签
            
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device).unsqueeze(1)  # (B, 1, H, W)
            
            # 前向传播
            if use_amp:
                with autocast():
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_targets).item()
                    # 使用111/metrics.py中的dice_coefficient函数
                    val_dice += dice_coefficient(val_outputs, val_targets)
            else:
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()
                # 使用111/metrics.py中的dice_coefficient函数
                val_dice += dice_coefficient(val_outputs, val_targets)
    
    # 计算平均值
    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    
    return avg_val_loss, avg_val_dice


# 计算验证损失和Dice（滑窗版本，用于patch训练模式）
def calculate_val_loss_and_dice_sliding(model, val_loader, device, patch_size=256, overlap=0.5, scaler=None):
    """
    使用滑窗预测计算验证集上的损失和Dice
    
    用于patch训练模式：验证时使用完整图像，但通过滑窗方式预测避免显存爆炸
    
    Args:
        model: 模型
        val_loader: 验证集DataLoader（完整图像）
        device: 设备
        patch_size: 滑窗patch尺寸
        overlap: 重叠比例（0.5表示50%重叠）
        scaler: GradScaler（可选）
    
    Returns:
        avg_val_loss: 验证损失
        avg_val_dice: 验证Dice
    """
    # 损失函数
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    
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
                gt = targets[i].to(device).long()  # (H, W)
                
                # 滑窗预测（使用重叠滑窗+高斯权重融合）
                prob_map, pred_mask = sliding_window_predict(
                    model, img, device,
                    patch_size=patch_size,
                    overlap=overlap
                )
                
                # 计算loss：使用prob_map和gt
                # 注意：prob_map已经是sigmoid后的概率，需要转回logits
                # 但为了简化，我们直接用BCE(prob, target)而不是BCEWithLogits
                # 或者直接基于pred_mask计算dice
                
                # 简化版：只计算Dice（因为loss需要logits，滑窗后难以准确计算）
                # 将pred_mask和gt展平计算Dice
                pred_flat = pred_mask.float()
                gt_flat = gt.float()
                
                # 计算Dice
                smooth = 1e-5
                intersection = (pred_flat * gt_flat).sum()
                dice = (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
                
                total_dice += dice.item()
                n += 1
                
                # Loss使用prob_map（需要转换为与BCE兼容的格式）
                # 使用prob_map直接计算（作为pseudo-loss）
                prob_map_expanded = prob_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                gt_expanded = gt.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
                
                # 计算BCE损失（注意：prob_map是sigmoid后的值，不能用BCEWithLogitsLoss）
                bce_val = F.binary_cross_entropy(prob_map_expanded, gt_expanded)
                
                # 计算Dice损失
                dice_loss_val = 1 - dice
                
                loss = 0.5 * bce_val + 0.5 * dice_loss_val
                total_loss += loss.item()
    
    avg_val_loss = total_loss / max(n, 1)
    avg_val_dice = total_dice / max(n, 1)
    
    return avg_val_loss, avg_val_dice


# 计算性能指标（完整版：准确率、Dice、Precision、Recall、IoU、F1、敏感性、特异性）
def evaluate(model, train_loader, test_loader, device, num_classes):
    """
    计算全面的评价指标（单通道二分类）
    
    Returns:
        train_acc, test_acc, train_metrics, test_metrics
    """
    model.eval()
    
    train_preds = []
    train_targets = []
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        # 训练集评估
        for train_image, train_target, _ in train_loader:
            train_image, train_target = train_image.to(device), train_target.to(device)
            train_output = model(train_image)  # (B, 1, H, W)
            # 单通道输出：sigmoid + 阈值0.5
            train_pred = (torch.sigmoid(train_output) > 0.5).squeeze(1).long()  # (B, H, W)
            
            train_preds.append(train_pred.cpu())
            train_targets.append(train_target.cpu())
        
        # 测试集评估
        for test_image, test_target, _ in test_loader:
            test_image, test_target = test_image.to(device), test_target.to(device)
            test_output = model(test_image)  # (B, 1, H, W)
            # 单通道输出：sigmoid + 阈值0.5
            test_pred = (torch.sigmoid(test_output) > 0.5).squeeze(1).long()  # (B, H, W)
            
            test_preds.append(test_pred.cpu())
            test_targets.append(test_target.cpu())
    
    # 拼接所有batch
    train_preds = torch.cat(train_preds, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    # 计算指标（使用metrics.py中的calculate_metrics）
    train_metrics = calculate_metrics(train_preds, train_targets, num_classes)
    test_metrics = calculate_metrics(test_preds, test_targets, num_classes)
    
    train_acc = train_metrics['accuracy']
    test_acc = test_metrics['accuracy']
    
    return train_acc, test_acc, train_metrics, test_metrics


# 可视化一个batch的图像和mask
def plot(data_loader):
    """可视化一个batch的数据"""
    # 从dataloader中取一个batch
    for imgs, masks, _ in data_loader:
        print(f"Image batch shape: {imgs.shape}")
        print(f"Mask batch shape: {masks.shape}")
        
        # 取第一个样本（高光谱图像，显示前3个波段作为RGB）
        img = imgs[0][:3].permute(1, 2, 0).numpy()  # 取前3个波段 (H, W, 3)
        mask = masks[0].numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title("Image (first 3 bands)")
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Mask")
        plt.savefig("data_visualization.png")
        print("Visualization saved to data_visualization.png")
        break


# 绘制学习率衰减曲线
def plot_lr_decay(scheduler, optimizer, epochs):
    """绘制学习率衰减曲线"""
    lr_list = []
    for epoch in range(epochs):
        lr_list.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), lr_list, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay Schedule')
    plt.grid(True, alpha=0.3)
    plt.savefig('lr_decay_schedule.png', dpi=150)
    plt.close()
    print("Learning rate decay schedule saved to lr_decay_schedule.png")
    
    # 重置scheduler
    for _ in range(epochs):
        optimizer.step()


# 绘制训练曲线（Loss + EMA Dice）
def plot_training_curves(train_loss_list, ema_dice_list, lr_list, fold_dir):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss曲线
    axes[0].plot(range(1, len(train_loss_list) + 1), train_loss_list, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # EMA Dice曲线
    axes[1].plot(range(1, len(ema_dice_list) + 1), ema_dice_list, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('EMA Dice', fontsize=12)
    axes[1].set_title('EMA Dice Score', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.3, 1])
    
    # 学习率曲线
    axes[2].plot(range(1, len(lr_list) + 1), lr_list, 'orange', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fold_dir}/training_curves.png', dpi=300)
    plt.close()


# 新增内容
import torch
import torch.nn.functional as F

def _reduce_to_1ch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, h, w) or (B, 1, h, w)
    return: (B, 1, h, w)
    """
    if x is None:
        return None
    if x.dim() != 4:
        return None
    if x.size(1) == 1:
        return x
    # 多通道注意力/特征 -> 做通道均值，得到单通道热力图
    return x.mean(dim=1, keepdim=True)

def sliding_window_predict_with_viz(
    model,
    image,                       # (C,H,W)  torch.Tensor
    device,
    patch_size=256,
    overlap=0.75,
    threshold=0.5,
    want_recon_err=True,
    want_A2=True,
    want_spga=True,
    want_pgac=True,
    want_A2_argmax=False,        # 默认关闭：论文不建议用
):
    """
    单fold整图滑窗推理 + 同步融合 SPGA/PGAC/recon/A2(原型无关指标)
    Returns:
        prob_map_fg: (H,W) CPU float32
        pred_mask:   (H,W) CPU int64
        viz: dict[str, CPU tensor(H,W)]
    """
    model.eval()
    image = image.to(device)

    C, H, W = image.shape
    ps = int(patch_size)
    stride = int(ps * (1 - overlap))
    if stride <= 0:
        stride = ps

    # ---- accumulators ----
    prob_acc = torch.zeros((H, W), dtype=torch.float32, device=device)
    w_acc    = torch.zeros((H, W), dtype=torch.float32, device=device)

    spga3_acc = spga4_acc = None
    pgac3_acc = pgac4_acc = pgac5_acc = None
    recon_acc = None

    A2_acc = None  # (K,H,W)
    A2_w   = None  # (H,W)

    # ---- Gaussian weight window ----
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, ps, device=device),
        torch.linspace(-1, 1, ps, device=device),
        indexing='ij'
    )
    dist = torch.sqrt(xx**2 + yy**2)
    gw = torch.exp(-(dist**2) / 0.5).float()  # (ps,ps)

    def _accum_2d(acc2d, patch2d, y0, x0):
        acc2d[y0:y0+ps, x0:x0+ps] += patch2d * gw

    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y0 = min(y, H - ps)
                x0 = min(x, W - ps)

                patch = image[:, y0:y0+ps, x0:x0+ps].unsqueeze(0)  # (1,C,ps,ps)

                out = None
                A2 = X2_hat = X2_down = None

                try:
                    out, A2, X2_hat, X2_down = model(patch, return_unmixing=True)
                except TypeError:
                    out = model(patch)

                prob_patch = torch.sigmoid(out)[0, 0]  # (ps,ps)
                prob_acc[y0:y0+ps, x0:x0+ps] += prob_patch * gw
                w_acc[y0:y0+ps, x0:x0+ps]    += gw

                # ---- SPGA ----
                if want_spga:
                    if hasattr(model, "spga3") and getattr(model.spga3, "last_attn_map", None) is not None:
                        attn = _reduce_to_1ch(model.spga3.last_attn_map)  # (1,1,h,w)
                        attn_up = F.interpolate(attn, size=(ps, ps), mode="bilinear", align_corners=False)[0, 0]
                        if spga3_acc is None:
                            spga3_acc = torch.zeros((H, W), dtype=torch.float32, device=device)
                        _accum_2d(spga3_acc, attn_up, y0, x0)

                    if hasattr(model, "spga4") and getattr(model.spga4, "last_attn_map", None) is not None:
                        attn = _reduce_to_1ch(model.spga4.last_attn_map)
                        attn_up = F.interpolate(attn, size=(ps, ps), mode="bilinear", align_corners=False)[0, 0]
                        if spga4_acc is None:
                            spga4_acc = torch.zeros((H, W), dtype=torch.float32, device=device)
                        _accum_2d(spga4_acc, attn_up, y0, x0)

                # ---- PGAC gate_spatial ----
                if want_pgac:
                    for name in ["pgac3", "pgac4", "pgac5"]:
                        if not hasattr(model, name):
                            continue
                        m = getattr(model, name)
                        gs = getattr(m, "last_gate_spatial", None)  # (1,1,h,w)
                        if gs is None:
                            continue
                        gs_up = F.interpolate(gs, size=(ps, ps), mode="bilinear", align_corners=False)[0, 0]

                        if name == "pgac3":
                            if pgac3_acc is None:
                                pgac3_acc = torch.zeros((H, W), dtype=torch.float32, device=device)
                            _accum_2d(pgac3_acc, gs_up, y0, x0)
                        elif name == "pgac4":
                            if pgac4_acc is None:
                                pgac4_acc = torch.zeros((H, W), dtype=torch.float32, device=device)
                            _accum_2d(pgac4_acc, gs_up, y0, x0)
                        else:
                            if pgac5_acc is None:
                                pgac5_acc = torch.zeros((H, W), dtype=torch.float32, device=device)
                            _accum_2d(pgac5_acc, gs_up, y0, x0)

                # ---- recon_err ----
                if want_recon_err and (X2_hat is not None) and (X2_down is not None):
                    err = (X2_hat - X2_down).abs().mean(dim=1, keepdim=True)  # (1,1,h,w)
                    err_up = F.interpolate(err, size=(ps, ps), mode="bilinear", align_corners=False)[0, 0]
                    if recon_acc is None:
                        recon_acc = torch.zeros((H, W), dtype=torch.float32, device=device)
                    _accum_2d(recon_acc, err_up, y0, x0)

                # ---- A2 (K-dim soft routing) ----
                if want_A2 and (A2 is not None):
                    # A2: (1,K,h,w)  -> up to (K,ps,ps)
                    A2_up = F.interpolate(A2, size=(ps, ps), mode="bilinear", align_corners=False)[0]  # (K,ps,ps)
                    K = A2_up.size(0)

                    if A2_acc is None:
                        A2_acc = torch.zeros((K, H, W), dtype=torch.float32, device=device)
                        A2_w   = torch.zeros((H, W), dtype=torch.float32, device=device)

                    # vectorized accumulate
                    A2_acc[:, y0:y0+ps, x0:x0+ps] += A2_up * gw.unsqueeze(0)
                    A2_w[y0:y0+ps, x0:x0+ps]      += gw

        # ---- normalize ----
        w_safe = torch.clamp(w_acc, min=1e-6)
        prob_map = prob_acc / w_safe
        pred_mask = (prob_map > threshold).long()

        viz = {}

        if spga3_acc is not None:
            viz["spga3_attn"] = (spga3_acc / w_safe).detach().cpu()
        if spga4_acc is not None:
            viz["spga4_attn"] = (spga4_acc / w_safe).detach().cpu()

        if pgac3_acc is not None:
            viz["pgac3_gate"] = (pgac3_acc / w_safe).detach().cpu()
        if pgac4_acc is not None:
            viz["pgac4_gate"] = (pgac4_acc / w_safe).detach().cpu()
        if pgac5_acc is not None:
            viz["pgac5_gate"] = (pgac5_acc / w_safe).detach().cpu()

        if recon_acc is not None:
            viz["recon_err"] = (recon_acc / w_safe).detach().cpu()

        # ---- A2: prototype-agnostic indicators ----
        if (A2_acc is not None) and (A2_w is not None):
            import math
            
            A2_w_safe = torch.clamp(A2_w, min=1e-6)
            A2_full = A2_acc / A2_w_safe.unsqueeze(0)  # (K,H,W)

            # ---- 关键：把A2当"权重"而不是"概率"，统一做非负+归一化，避免数值/符号导致统计图怪异 ----
            A2_pos = torch.clamp(A2_full, min=0.0)
            A2_sum = A2_pos.sum(dim=0, keepdim=True).clamp_min(1e-8)
            A2p = A2_pos / A2_sum  # (K,H,W) 每个像素K通道和为1

            K = A2p.size(0)
            eps = 1e-8

            # top1 / top2
            top2v, top2i = torch.topk(A2p, k=2, dim=0)   # (2,H,W)
            conf = top2v[0]                               # (H,W) top1概率
            margin = (top2v[0] - top2v[1])                # (H,W) top1-top2

            # entropy：给你"归一化熵"，范围[0,1]，更适合统一颜色刻度
            H_raw = -(A2p * (A2p + eps).log()).sum(dim=0)   # (H,W)
            H_norm = H_raw / math.log(K)                    # (H,W) 0~1

            # effk：有效原型数，范围[1,K]
            effk = torch.exp(H_raw)                         # (H,W)

            # 可选：argmax仍然保留用于诊断（但不再作为核心证据）
            argmax = torch.argmax(A2p, dim=0)               # (H,W)

            viz["A2_conf"] = conf.detach().cpu()
            viz["A2_margin"] = margin.detach().cpu()
            viz["A2_entropy"] = H_norm.detach().cpu()
            viz["A2_effk"] = effk.detach().cpu()

            if want_A2_argmax:
                viz["A2_argmax"] = argmax.detach().cpu()

            # 你要是还想保留proto热力图，也建议存A2p而不是A2_full
            for k in range(K):
                viz[f"A2_proto{k}"] = A2p[k].detach().cpu()

    return prob_map.detach().cpu(), pred_mask.detach().cpu(), viz

