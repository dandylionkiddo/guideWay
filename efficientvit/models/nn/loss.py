import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1.0):
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
        targets_one_hot = F.one_hot(targets, num_classes=self.n_classes).permute(0, 3, 1, 2).float()

        # Flatten input and target tensors
        inputs = inputs.contiguous().view(inputs.shape[0], self.n_classes, -1)
        targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.shape[0], self.n_classes, -1)

        # Calculate intersection and union
        intersection = torch.sum(inputs * targets_one_hot, dim=2)
        union = torch.sum(inputs, dim=2) + torch.sum(targets_one_hot, dim=2)

        # Calculate Dice score and Dice loss
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss.mean()

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
