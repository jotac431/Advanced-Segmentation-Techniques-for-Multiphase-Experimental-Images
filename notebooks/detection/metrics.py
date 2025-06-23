import torch

def compute_dice(preds, targets, num_classes):
    smooth = 1e-6
    preds = torch.argmax(preds, dim=1)  # shape: (B, H, W)
    dice_scores = []

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    return sum(dice_scores) / num_classes
