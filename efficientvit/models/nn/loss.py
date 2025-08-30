import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) - model output (logits)
        targets: (N, H, W) - ground truth labels
        """
        # Apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)

        # One-hot encode the target
        targets_one_hot = F.one_hot(targets, num_classes=self.n_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # 각 클래스별로 독립적으로 Dice 계산
        dice_per_class = []

        for c in range(self.n_classes):
            # 해당 클래스에 대한 예측과 타겟
            input_c = inputs[:, c, :, :]  # (N, H, W)
            target_c = targets_one_hot[:, c, :, :]  # (N, H, W)

            # Batch 전체에 대해 계산 (중요!)
            intersection = torch.sum(input_c * target_c)
            cardinality = torch.sum(input_c) + torch.sum(target_c)

            dice_c = (2. * intersection + self.smooth) / (cardinality + self.smooth)
            dice_per_class.append(dice_c)

        # 모든 클래스의 평균 Dice
        mean_dice = torch.stack(dice_per_class).mean()

        return 1 - mean_dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) - model output (logits)
        targets: (N, H, W) - ground truth labels
        """
        # Calculate Cross Entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')

        # Get probabilities of the correct class
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ComboLoss(nn.Module):
    """
    Focal Loss와 Dice Loss를 조합한 복합 손실 함수입니다.
    두 손실 함수의 가중치를 조절하여 사용할 수 있습니다.
    """
    def __init__(self, n_classes, focal_alpha=0.25, focal_gamma=2.0, focal_weight=1.0, dice_smooth=1e-6, dice_weight=1.0, ignore_index=255):
        super(ComboLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        # FocalLoss와 DiceLoss 인스턴스를 생성합니다.
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        self.dice_loss = DiceLoss(
            n_classes=n_classes,
            smooth=dice_smooth
        )

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) - 모델 출력 (logits)
        targets: (N, H, W) - 실제 레이블
        """
        # 각 손실을 계산합니다.
        loss_focal = self.focal_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        
        # 가중치를 적용하여 최종 손실을 계산합니다.
        combined_loss = (self.focal_weight * loss_focal) + (self.dice_weight * loss_dice)
        
        return combined_loss