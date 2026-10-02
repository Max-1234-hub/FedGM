# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:11:42 2025

@author: ZML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:11:42 2025

@author: ZML
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:11:42 2025

@author: ZML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """Dice Loss + BCE Loss Combination"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        """
        Args:
            predict: (B, C, H, W) or (B, C, D, H, W) logits
            target: (B, H, W) or (B, D, H, W)
        """
        num_classes = predict.shape[1]
        if num_classes == 1:
            # BCE Loss
            ce = self.bce_loss(predict, target.float().unsqueeze(1))
            # Dice Loss
            predict = torch.sigmoid(predict).squeeze(1)
            if predict.dim() == 3:
                intersection = torch.sum(predict * target, dim=(1, 2))
                union = torch.sum(predict, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
            if predict.dim() == 4:
                intersection = torch.sum(predict * target, dim=(1, 2, 3))
                union = torch.sum(predict, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
        else:
            # CE Loss
            ce = self.ce_loss(predict, target.long())
            # Dice Loss
            predict = torch.softmax(predict, dim=1)
            if predict.dim() == 4:
                target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
                intersection = torch.sum(predict * target_one_hot, dim=(2, 3))
                union = torch.sum(predict, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
            if predict.dim() == 5:
                target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()
                intersection = torch.sum(predict * target_one_hot, dim=(2, 3, 4))
                union = torch.sum(predict, dim=(2, 3, 4)) + torch.sum(target_one_hot, dim=(2, 3, 4))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return self.ce_weight * ce + self.dice_weight * dice_loss
    
    
class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        num_classes = predict.shape[1]
        
        if target.dim() == 4:
            target = target.squeeze(1)
            
        ce = self.ce_loss(predict, target.long())
        
        predict_softmax = torch.softmax(predict, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
            
        intersection = torch.sum(predict_softmax * target_one_hot, dim=(2, 3))
        union = torch.sum(predict_softmax, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return self.ce_weight * ce + self.dice_weight * dice_loss
    
    
class DiceSoftmaxLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(DiceSoftmaxLoss, self).__init__()  # Fixed super call
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        if target.dim() == 4:
            target = target.squeeze(1)
        
        bce = self.ce_loss(predict, target.long())
        predict_softmax = torch.softmax(predict, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=2).permute(0, 3, 1, 2).float()
            
        intersection = (predict_softmax * target_one_hot).sum(dim=(2, 3))
        union = predict_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_per_class.mean(dim=0)
        
        return self.bce_weight * bce + self.dice_weight * dice_loss
    
    
class BinaryKLLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(BinaryKLLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, student_logits, teacher_logits):
        student_log_probs = F.logsigmoid(student_logits / self.temperature)
        student_log_neg_probs = F.logsigmoid(-student_logits / self.temperature)
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        
        kl_loss = teacher_probs * (torch.log(teacher_probs.clamp(1e-8)) - student_log_probs) + \
                 (1 - teacher_probs) * (torch.log((1 - teacher_probs).clamp(1e-8)) - student_log_neg_probs)

        return kl_loss.mean() * (self.temperature ** 2)
    
    
class BinaryDistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4.0):
        super(BinaryDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.dice_bce_loss = DiceBCELoss()
        self.kl_loss = BinaryKLLoss(temperature)
        
    def forward(self, student_logits, teacher_logits, labels):
        bce_loss = self.dice_bce_loss(student_logits, labels.float())
        distill_loss = self.kl_loss(student_logits, teacher_logits)
        total_loss = (1 - self.alpha) * bce_loss + self.alpha * distill_loss
        return total_loss, bce_loss, distill_loss


# --- ADDED MISSING CLASSES BELOW ---

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):
    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss, loss1, loss2


class FocalDiceloss(nn.Module):
    def __init__(self, weight=20.0):
        super(FocalDiceloss, self).__init__()
        self.weight = weight
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        loss = self.weight * focal_loss + dice_loss
        return loss
