# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),

            # With batch_size = 2, batch stats (especially BatchNorm) may be unstable.
            # Use GroupNorm instead of BatchNorm,
            # First test was with BatchNorm and got low mIoU
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),

            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.middle = ConvBlock(512, 1024)

        self.up4 = UpBlock(1024, 512, 512)
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        m = self.middle(F.max_pool2d(e4, 2))

        d4 = self.up4(m, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.out_conv(d1)


class UNetWithLoss(nn.Module):
    def __init__(self, num_classes, loss_fn=None):
        super().__init__()
        self.model = UNet(num_classes=num_classes)
        
        # Compute class frequency in dataset beforehand
        #weights = torch.tensor([0.2, 0.8], device=device)  # Example for [background, bubble]
        #self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def dice_loss(self, logits, targets, eps=1e-6):
        """
        Compute Dice loss from logits and integer targets.
        Assumes logits are raw (not softmaxed), and targets are integer-encoded.

        logits: Tensor of shape [B, C, H, W]
        targets: Tensor of shape [B, H, W] with values in [0, C-1]
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dim=dims)
        cardinality = (probs + targets_onehot).sum(dim=dims)
        dice = (2. * intersection + eps) / (cardinality + eps)
        return 1 - dice.mean()

    def forward(self, images, targets=None):
        if isinstance(images, (list, tuple)):
            x = torch.stack(images)
        else:
            x = images

        logits = self.model(x)

        if self.training:
            # Build pixel-wise label map
            target_masks = torch.zeros_like(logits[:, 0], dtype=torch.long)
            for i, t in enumerate(targets):
                for mask, label in zip(t["masks"], t["labels"]):
                    target_masks[i] = torch.logical_or(target_masks[i], mask.bool()).long()

            ce_loss = self.loss_fn(logits, target_masks)
            dice_loss = self.dice_loss(logits, target_masks)
            return {"loss": ce_loss + dice_loss}
        else:
            preds = torch.argmax(logits, dim=1)
            results = []
            for i in range(preds.size(0)):
                mask = preds[i].detach().cpu()
                results.append({
                    "masks": mask.unsqueeze(0).float(),
                    "scores": torch.tensor([1.0]),
                    "labels": torch.tensor([1])
                })
            return results


def get_unet(num_classes):
    return UNetWithLoss(num_classes=num_classes)

