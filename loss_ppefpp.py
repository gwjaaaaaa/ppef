"""
Loss Functions for PPEF++
PPEF++框架的损失函数

包含：
1. 分割损失（Dice + BCE）
2. 光谱重建损失（MSE）
3. 丰度图平滑损失（TV Loss）
4. 原型多样性损失（Diversity Loss）
5. 丰度熵损失（Entropy Loss）
6. 原型正交约束（Orthogonality Loss）【新增】
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPEFPPLoss(nn.Module):
    """
    PPEF++完整损失函数
    
    Total Loss = λ_seg * L_seg 
                + λ_recon * L_recon
                + λ_smooth * L_smooth
                + λ_div * L_div
                + λ_entropy * L_entropy
                + λ_orth * L_orth  【新增】
    
    Args:
        lambda_recon: 光谱重建损失权重（默认0.1）
        lambda_smooth: 丰度图平滑损失权重（默认0.01）
        lambda_div: 原型多样性损失权重（默认0.01）
        lambda_entropy: 丰度熵损失权重（默认0.001）
        lambda_orth: 原型正交约束权重（默认0.01）【新增】
    """
    
    def __init__(self,
                 lambda_recon=0.03,     # 原来 0.1，降低重建任务权重
                 lambda_smooth=0.005,   # 原来 0.01
                 lambda_div=0.02,       # 原来 0.01
                 lambda_entropy=0.005,  # 原来 0.001
                 lambda_orth=0.01,      # 【新增】原型正交约束
                 warmup_epochs=10):     # 前10轮只优化分割
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_smooth = lambda_smooth
        self.lambda_div = lambda_div
        self.lambda_entropy = lambda_entropy
        self.lambda_orth = lambda_orth  # 【新增】
        self.warmup_epochs = warmup_epochs
        
        # 分割损失（Dice + BCE）
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        print(f"[PPEFPPLoss] Initialized with:")
        print(f"  λ_recon: {lambda_recon}")
        print(f"  λ_smooth: {lambda_smooth}")
        print(f"  λ_div: {lambda_div}")
        print(f"  λ_entropy: {lambda_entropy}")
        print(f"  λ_orth: {lambda_orth}  【新增】原型正交约束")
        print(f"  warmup_epochs: {warmup_epochs} (seg-only for first {warmup_epochs} epochs)")
    
    def forward(self, 
                pred, target,
                A2=None, X2_hat=None, X2_down=None, 
                prototypes=None,
                epoch=None):
        """
        计算总损失
        
        Args:
            pred: (B, 1, H, W) 分割预测（logits）
            target: (B, H, W) 分割标签
            A2: (B, K, H, W) 丰度图（可选）
            X2_hat: (B, num_bands, H, W) 重建光谱（可选）
            X2_down: (B, num_bands, H, W) 真实光谱（可选）
            prototypes: (K, num_bands) 光谱原型（可选）
            epoch: 当前epoch（用于warmup，可选）
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # ========== 1. 分割损失 ==========
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target.unsqueeze(1).float())
        seg_loss = 0.5 * dice_loss + 0.5 * bce_loss  # ✅ 改为加权平均，与验证保持一致
        
        loss_dict = {
            'seg': seg_loss.detach(),
            'dice': dice_loss.detach(),
            'bce': bce_loss.detach(),
        }
        
        total_loss = seg_loss
        
        # ========== Warmup：前warmup_epochs只优化分割 ==========
        if epoch is not None and epoch < self.warmup_epochs:
            loss_dict['total'] = total_loss.detach()
            loss_dict['warmup'] = True
            return total_loss, loss_dict
        
        # ========== 2. 光谱重建损失（如果提供）==========
        if X2_hat is not None and X2_down is not None:
            recon_loss = F.mse_loss(X2_hat, X2_down)
            total_loss = total_loss + self.lambda_recon * recon_loss
            loss_dict['recon'] = recon_loss.detach()
        
        # ========== 3. 丰度图平滑损失（如果提供）==========
        if A2 is not None:
            smooth_loss = self.tv_loss(A2)
            total_loss = total_loss + self.lambda_smooth * smooth_loss
            loss_dict['smooth'] = smooth_loss.detach()
        
        # ========== 4. 原型多样性损失（如果提供）==========
        if prototypes is not None:
            div_loss = self.diversity_loss(prototypes)
            total_loss = total_loss + self.lambda_div * div_loss
            loss_dict['div'] = div_loss.detach()
        
        # ========== 5. 丰度熵损失（如果提供）==========
        if A2 is not None:
            entropy_loss = self.entropy_loss(A2)
            total_loss = total_loss + self.lambda_entropy * entropy_loss
            loss_dict['entropy'] = entropy_loss.detach()
        
        # ========== 6. 原型正交约束（如果提供）【新增】==========
        if prototypes is not None:
            orth_loss = self.orthogonality_loss(prototypes)
            total_loss = total_loss + self.lambda_orth * orth_loss
            loss_dict['orth'] = orth_loss.detach()
        
        loss_dict['total'] = total_loss.detach()
        
        return total_loss, loss_dict
    
    def tv_loss(self, A: torch.Tensor, eps=1e-8):
        """
        Total Variation Loss - 促进丰度图空间平滑
        
        Args:
            A: (B, K, H, W) 丰度图
            eps: 防止除零
        
        Returns:
            tv_loss: TV损失
        """
        # 水平方向的变化
        tv_h = torch.abs(A[:, :, 1:, :] - A[:, :, :-1, :]).mean()
        
        # 垂直方向的变化
        tv_w = torch.abs(A[:, :, :, 1:] - A[:, :, :, :-1]).mean()
        
        return tv_h + tv_w
    
    def diversity_loss(self, prototypes: torch.Tensor, eps=1e-8):
        """
        Prototype Diversity Loss - 促进原型多样性
        
        Args:
            prototypes: (K, num_bands) 光谱原型
            eps: 防止除零
        
        Returns:
            div_loss: 多样性损失（相似度越小越好）
        """
        # 归一化原型
        P_norm = F.normalize(prototypes, dim=1, eps=eps)  # (K, num_bands)
        
        # 计算两两相似度矩阵
        similarity = torch.matmul(P_norm, P_norm.T)  # (K, K)
        
        # 去掉对角线（自己与自己的相似度）
        K = similarity.shape[0]
        mask = torch.eye(K, device=similarity.device)
        similarity = similarity * (1 - mask)
        
        # 损失：原型间相似度越小越好
        # 这里取相似度的绝对值的均值作为损失
        div_loss = similarity.abs().sum() / (K * (K - 1) + eps)
        
        return div_loss
    
    def entropy_loss(self, A: torch.Tensor, eps=1e-8):
        """
        Abundance Entropy Loss - 促进丰度图稀疏性
        
        鼓励每个像素只激活少数几个原型（不是均匀分布）
        
        Args:
            A: (B, K, H, W) 丰度图
            eps: 防止log(0)
        
        Returns:
            entropy_loss: 熵损失（熵越小越好，即分布越集中）
        """
        # 计算每个像素的熵
        # H(A) = -∑(A_k * log(A_k))
        entropy = -torch.sum(A * torch.log(A + eps), dim=1)  # (B, H, W)
        
        # 平均熵
        avg_entropy = entropy.mean()
        
        # 损失：熵越小越好（希望分布集中）
        return avg_entropy
    
    def orthogonality_loss(self, prototypes: torch.Tensor, eps=1e-8):
        """
        Prototype Orthogonality Loss - 原型正交约束【新增】
        
        强制学习到的光谱原型互不相关，使得生成的丰度图解耦更彻底，对比度更高。
        通过最小化归一化原型间的相关矩阵与单位矩阵的Frobenius范数来实现。
        
        Args:
            prototypes: (K, num_bands) 光谱原型
            eps: 防止除零
        
        Returns:
            orth_loss: 正交损失（相关性越小越好）
        """
        K = prototypes.shape[0]
        
        # 归一化原型（L2归一化）
        P_norm = F.normalize(prototypes, p=2, dim=1, eps=eps)  # (K, num_bands)
        
        # 计算相关矩阵 P_norm @ P_norm^T
        correlation_matrix = torch.matmul(P_norm, P_norm.T)  # (K, K)
        
        # 目标：相关矩阵应该接近单位矩阵（对角线为1，非对角线为0）
        identity = torch.eye(K, device=prototypes.device, dtype=prototypes.dtype)
        
        # Frobenius范数的平方：||C - I||_F^2
        orth_loss = torch.norm(correlation_matrix - identity, p='fro') ** 2
        
        # 归一化（除以元素总数）
        orth_loss = orth_loss / (K * K)
        
        return orth_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) labels {0, 1}
        """
        pred = torch.sigmoid(pred)  # (B, 1, H, W)
        pred = pred.squeeze(1)      # (B, H, W)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1).float()
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss
        return 1.0 - dice


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing PPEF++ Loss Functions")
    print("="*70)
    
    # 测试参数
    B = 2
    H, W = 64, 64
    num_bands = 40
    num_proto = 4
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {B}")
    print(f"  Spatial size: ({H}, {W})")
    print(f"  Spectral bands: {num_bands}")
    print(f"  Num prototypes: {num_proto}")
    
    # 创建损失函数
    print(f"\n{'='*70}")
    criterion = PPEFPPLoss(
        lambda_recon=0.1,
        lambda_smooth=0.01,
        lambda_div=0.01,
        lambda_entropy=0.001
    )
    print(f"{'='*70}")
    
    # 创建模拟数据
    pred = torch.randn(B, 1, H, W)  # 分割预测（logits）
    target = torch.randint(0, 2, (B, H, W))  # 分割标签 {0, 1}
    
    A2 = F.softmax(torch.randn(B, num_proto, H, W), dim=1)  # 丰度图（归一化）
    X2_hat = torch.randn(B, num_bands, H, W)  # 重建光谱
    X2_down = torch.randn(B, num_bands, H, W)  # 真实光谱
    prototypes = torch.randn(num_proto, num_bands)  # 光谱原型
    
    print(f"\nInput shapes:")
    print(f"  pred: {pred.shape}")
    print(f"  target: {target.shape}")
    print(f"  A2: {A2.shape}")
    print(f"  X2_hat: {X2_hat.shape}")
    print(f"  X2_down: {X2_down.shape}")
    print(f"  prototypes: {prototypes.shape}")
    
    # 验证丰度图性质
    A2_sum = A2.sum(dim=1)
    print(f"\nAbundance map check:")
    print(f"  Sum: mean={A2_sum.mean():.4f}, std={A2_sum.std():.4f}")
    
    # 计算损失
    print(f"\n{'='*70}")
    print("Computing losses...")
    print(f"{'='*70}")
    
    total_loss, loss_dict = criterion(
        pred, target,
        A2=A2,
        X2_hat=X2_hat,
        X2_down=X2_down,
        prototypes=prototypes
    )
    
    print(f"\nLoss breakdown:")
    for name, value in loss_dict.items():
        print(f"  {name:15s}: {value.item():.6f}")
    
    # 测试只有分割损失的情况
    print(f"\n{'='*70}")
    print("Testing seg-only loss (no PPEF++ components)...")
    print(f"{'='*70}")
    
    seg_only_loss, seg_only_dict = criterion(pred, target)
    
    print(f"\nLoss breakdown (seg only):")
    for name, value in seg_only_dict.items():
        print(f"  {name:15s}: {value.item():.6f}")
    
    # 梯度测试
    print(f"\n{'='*70}")
    print("Gradient test...")
    print(f"{'='*70}")
    
    pred_test = torch.randn(1, 1, H, W, requires_grad=True)
    target_test = torch.randint(0, 2, (1, H, W))
    A2_test = F.softmax(torch.randn(1, num_proto, H, W, requires_grad=True), dim=1)
    X2_hat_test = torch.randn(1, num_bands, H, W, requires_grad=True)
    X2_down_test = torch.randn(1, num_bands, H, W)
    prototypes_test = torch.randn(num_proto, num_bands, requires_grad=True)
    
    loss, _ = criterion(
        pred_test, target_test,
        A2=A2_test,
        X2_hat=X2_hat_test,
        X2_down=X2_down_test,
        prototypes=prototypes_test
    )
    
    loss.backward()
    
    print(f"✓ Backward pass successful")
    print(f"✓ pred gradient: {pred_test.grad is not None}")
    print(f"✓ A2 gradient: {A2_test.grad is not None}")
    print(f"✓ X2_hat gradient: {X2_hat_test.grad is not None}")
    print(f"✓ prototypes gradient: {prototypes_test.grad is not None}")
    
    # 测试各个损失项
    print(f"\n{'='*70}")
    print("Testing individual loss components...")
    print(f"{'='*70}")
    
    # TV Loss
    tv = criterion.tv_loss(A2)
    print(f"  TV Loss: {tv.item():.6f}")
    
    # Diversity Loss
    div = criterion.diversity_loss(prototypes)
    print(f"  Diversity Loss: {div.item():.6f}")
    
    # Entropy Loss
    entropy = criterion.entropy_loss(A2)
    print(f"  Entropy Loss: {entropy.item():.6f}")
    
    # Orthogonality Loss 【新增】
    orth = criterion.orthogonality_loss(prototypes)
    print(f"  Orthogonality Loss: {orth.item():.6f}")
    
    # Dice Loss
    dice_criterion = DiceLoss()
    dice = dice_criterion(pred, target)
    print(f"  Dice Loss: {dice.item():.6f}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)

