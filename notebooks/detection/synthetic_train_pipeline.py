# synthetic_train_pipeline.py

# =====================================================
# Imports
# =====================================================
import os
import sys
import time
import csv
import random
import yaml
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torchmetrics import JaccardIndex, Accuracy

# Local imports
from models import MODEL_REGISTRY
import utils
from engine import train_one_epoch, evaluate


# =====================================================
# Reproducibility
# =====================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================
# Helper: flatten config dictionary for TensorBoard
# =====================================================
def flatten_dict(d, parent_key: str = "", sep: str = "/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# =====================================================
# COCO-style Dataset Definition
# =====================================================
class COCOSegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None, subset_fraction: float = 1.0):
        self.root_dir = root_dir
        self.transforms = transforms

        self.ann_path = os.path.join(root_dir, "_annotations.coco.json")
        self.coco = COCO(self.ann_path)
        all_ids = list(self.coco.imgs.keys())

        subset_len = int(len(all_ids) * subset_fraction)
        self.ids = all_ids[:subset_len]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        img = read_image(img_path)
        img = tv_tensors.Image(img)
        h, w = img.shape[-2:]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        masks, boxes, labels, areas, iscrowd = [], [], [], [], []

        for ann in anns:
            if 'segmentation' not in ann:
                continue

            rles = coco_mask.frPyObjects(ann['segmentation'], h, w)
            mask = coco_mask.decode(rles)
            if mask.ndim == 3:
                mask = mask.any(axis=2)
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            masks.append(mask)

            x, y, bw, bh = ann['bbox']
            boxes.append(torch.tensor([x, y, x + bw, y + bh], dtype=torch.float32))
            labels.append(ann.get('category_id', 1))
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        if masks:
            masks = torch.stack(masks)
            boxes = torch.stack(boxes)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

            # Filter invalid boxes AFTER transforms
            boxes = target["boxes"]
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

            target["boxes"] = tv_tensors.BoundingBoxes(
                boxes[keep], format="XYXY", canvas_size=img.shape[-2:]
            )
            target["labels"] = target["labels"][keep]

            if "masks" in target:
                target["masks"] = target["masks"][keep]
            if "area" in target:
                target["area"] = target["area"][keep]
            if "iscrowd" in target:
                target["iscrowd"] = target["iscrowd"][keep]

        return img, target

    def __len__(self):
        return len(self.ids)


# -------------------------------------------------------------------------
# Dynamic Augmentation Parser
# -------------------------------------------------------------------------
def build_augmentations(cfg, train=True):
    aug = cfg["augmentation"]
    model_name = cfg["model"]["type"]
    is_unet = "unet" in model_name.lower()

    t_list = []

    # ----------------------------------------------------------
    # Resize always first
    # ----------------------------------------------------------
    t_list.append(T.Resize((512, 512)))

    if train:
        if aug.get("strong_color_jitter", False):
            t_list.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))

        if aug.get("intensity_shift", False):
            t_list.append(T.RandomAdjustSharpness(sharpness_factor=2, p=0.3))

        t_list.append(T.RandomHorizontalFlip(p=0.5))

        if aug.get("blur", False):
            t_list.append(T.GaussianBlur(kernel_size=3))

        # IMPORTANT: Rotation disabled for Mask R-CNN
        if aug.get("rotation", False) and is_unet:
            # Only allowed for UNet
            t_list.append(T.RandomRotation(10, fill=0))

    # Always convert at the end
    t_list += [
        T.ToDtype(torch.float32, scale=True),
        T.ToPureTensor(),
    ]

    return T.Compose(t_list)


# =====================================================
# Backbone Freezing Utilities
# =====================================================
def freeze_backbone(model):
    if hasattr(model, "backbone"):
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False

        # Freeze BatchNorm always
        for m in model.backbone.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        print("🔒 Backbone frozen (BN disabled).")



def unfreeze_backbone(model):
    if hasattr(model, "backbone"):

        # Allow backbone convs to train
        for name, param in model.backbone.named_parameters():
            param.requires_grad = True

        # BUT keep BatchNorm frozen
        for m in model.backbone.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                print(m.training, m.weight.requires_grad)
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        print("🔓 Backbone unfrozen (BN still frozen).")



# =====================================================
# Build Dataloaders
# =====================================================
def build_dataloaders(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = COCOSegmentationDataset(cfg["dataset"]["train"], build_augmentations(cfg, True))
    val_ds   = COCOSegmentationDataset(cfg["dataset"]["val"],   build_augmentations(cfg, False))
    test_ds  = COCOSegmentationDataset(cfg["dataset"]["test"],  build_augmentations(cfg, False))

    workers = cfg["train"]["num_workers"] if torch.cuda.is_available() else 0

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=workers,
        collate_fn=utils.collate_fn,
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=utils.collate_fn,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=utils.collate_fn,
    )

    return device, train_dl, val_dl, test_dl


# =====================================================
# Main Training Function
# =====================================================
def run_synthetic_training(cfg):
    set_seed(cfg["experiment"]["seed"])

    # Build dataloaders
    device, train_dl, val_dl, test_dl = build_dataloaders(cfg)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = cfg["model"]
    model = MODEL_REGISTRY[model_cfg["type"]](model_cfg["num_classes"])
    model.to(device)

    # ------------------------------------------------------------------
    # Optimizer with backbone LR factor
    # ------------------------------------------------------------------
    lr = cfg["train"]["lr"]
    backbone_factor = cfg["train"]["backbone_lr_factor"]

    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.SGD(
        [
            {"params": head_params, "lr": lr},
            {"params": backbone_params, "lr": lr * backbone_factor},
        ],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg["train"]["lr_step_size"], 
        gamma=cfg["train"]["gamma"]
    )

    # AMP scaler
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = cfg["experiment"]["name"]
    out_dir = os.path.join(cfg["output"]["dir"], f"{exp_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(cfg["logging"]["log_dir"], f"{exp_name}_{timestamp}"))
    writer.add_text("config", f"```yaml\n{yaml.dump(cfg, sort_keys=False)}\n```")

    # Track best metric / early stopping
    best_val = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    patience = 5

    model_name = cfg["model"]
    is_unet = "unet" in model_name
    num_epochs = cfg["train"]["epochs"]

    # CSV paths
    csv_path = os.path.join(out_dir, f"{exp_name}_metrics.csv")

    print(f"\n======== Starting Synthetic Training: {exp_name} ========\n")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg['train']['epochs']}")

        print("LRs:", [g["lr"] for g in optimizer.param_groups])

        # Train one epoch
        metric_logger = train_one_epoch(
            model,
            optimizer,
            train_dl,
            device,
            epoch,
            print_freq=cfg["train"]["print_freq"],
            scaler=scaler,
        )

        scheduler.step()

        # Log train metrics
        for name, meter in metric_logger.meters.items():
            writer.add_scalar(f"Train/{name}", meter.global_avg, epoch)

        # =====================================================
        # Validation
        # =====================================================
        print("Validating...")

        if is_unet:
            # ========================
            # UNet: Semantic metrics
            # ========================

            model.eval()

            miou_metric = JaccardIndex(
                num_classes=cfg["model"]["num_classes"], average="macro", task="multiclass"
            ).to(device)
            acc_metric = Accuracy(
                num_classes=cfg["model"]["num_classes"], average="macro", task="multiclass"
            ).to(device)

            all_preds = []
            all_targets = []

            with torch.no_grad():
                for images, targets in val_dl:
                    images = (
                        images.to(device)
                        if isinstance(images, torch.Tensor)
                        else torch.stack(images).to(device)
                    )

                    target_masks = torch.stack(
                        [t["masks"].sum(dim=0).clamp(0, 1).long() for t in targets]
                    ).to(device)

                    outputs = model(images)
                    preds = torch.stack([r["masks"].squeeze(0) for r in outputs]).to(
                        device
                    )

                    if preds.shape[-2:] != target_masks.shape[-2:]:
                        target_masks = torch.nn.functional.interpolate(
                            target_masks.unsqueeze(1).float(),
                            size=preds.shape[-2:],
                            mode="nearest",
                        ).squeeze(1).long()

                    all_preds.append(preds)
                    all_targets.append(target_masks)

                    miou_metric.update(preds, target_masks)
                    acc_metric.update(preds, target_masks)

            # Compute metrics
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            dice_score = torch.tensor(0.0)  # optional, can implement later

            miou = miou_metric.compute().item()
            pixel_acc = acc_metric.compute().item()

            print(f"UNet Validation — mIoU={miou:.4f}, Pixel Acc={pixel_acc:.4f}")

            writer.add_scalar("Val/mIoU", miou, epoch)
            writer.add_scalar("Val/PixelAcc", pixel_acc, epoch)

            current_val = miou

            # --- UNet CSV ---
            write_header = (not os.path.exists(csv_path))
            with open(csv_path, mode="a", newline="") as f:
                writer_csv = csv.writer(f)
                if write_header:
                    writer_csv.writerow(["epoch", "mIoU", "PixelAccuracy"])
                writer_csv.writerow([epoch, miou, pixel_acc])

        else:
            # ========================
            # Mask R-CNN: COCO metrics
            # ========================
            coco_eval = evaluate(model, val_dl, device=device)
            segm_stats = coco_eval.coco_eval["segm"].stats

            writer.add_scalar("Val/mAP", segm_stats[0], epoch)
            writer.add_scalar("Val/mAP50", segm_stats[1], epoch)
            writer.add_scalar("Val/mAP75", segm_stats[2], epoch)

            current_val = segm_stats[0]  # mAP@[.50:.95]

            # --- Mask R-CNN CSV ---
            write_header = (not os.path.exists(csv_path))
            with open(csv_path, mode="a", newline="") as f:
                writer_csv = csv.writer(f)
                if write_header:
                    writer_csv.writerow(
                        [
                            "epoch",
                            "AP@[0.50:0.95]",
                            "AP@0.50",
                            "AP@0.75",
                            "AP_small",
                            "AP_medium",
                            "AP_large",
                            "AR@1",
                            "AR@10",
                            "AR@100",
                            "AR_small",
                            "AR_medium",
                            "AR_large",
                        ]
                    )
                writer_csv.writerow([epoch] + list(segm_stats))

        # =====================================================
        # Early Stopping & Best Checkpoint
        # =====================================================
        if current_val > best_val:
            print(f"New best value: {current_val} (previous {best_val})")
            best_val = current_val
            best_epoch = epoch + 1
            epochs_no_improve = 0

            torch.save(
                model.state_dict(),
                os.path.join(out_dir, "model_best.pth"),
            )
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # =====================================================
    # Summary
    # =====================================================
    print(f"Training finished. Best epoch={best_epoch}, Best value={best_val}")

    writer.add_hparams(
        hparam_dict=flatten_dict(cfg),
        metric_dict={"best_val_metric": best_val},
    )
    writer.close()

    return model, best_val, best_epoch, cfg
        

# =====================================================
# CLI Entry Point
# =====================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_synthetic_training(cfg)
