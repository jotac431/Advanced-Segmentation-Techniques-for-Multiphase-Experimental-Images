# synthetic_test_eval.py
# Improved, future-proof evaluator for Mask R-CNN / UNet
# - Structured EvalResults
# - JSON summary logging
# - Deterministic visual examples
# - Per-image CSV + plots-ready outputs
# - Semantic metrics + bubble count + area stats
# - Fully notebook-friendly

import os
import time
import csv
import json
import random
import numpy as np
import torch
from dataclasses import dataclass, asdict

from torchvision.io import read_image
from torchvision import tv_tensors
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

import utils
from synthetic_train_pipeline import build_augmentations


# ============================================================
# Structured result container
# ============================================================
@dataclass
class EvalResults:
    semantic_metrics: dict
    instance_metrics: dict
    timing: dict
    config_used: dict
    examples_dir: str

    def to_dict(self):
        return asdict(self)


# ============================================================
# COCO Segmentation Dataset (masks only)
# ============================================================
class COCOSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        ann_path = os.path.join(root_dir, "_annotations.coco.json")
        self.coco = COCO(ann_path)
        self.ids = sorted(list(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        img_info = coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.root_dir, img_info["file_name"])
        img = read_image(img_path)
        img = tv_tensors.Image(img)
        h, w = img.shape[-2:]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        masks = []
        for ann in anns:
            if "segmentation" not in ann:
                continue
            rles = coco_mask.frPyObjects(ann["segmentation"], h, w)
            mask = coco_mask.decode(rles)
            if mask.ndim == 3:
                mask = mask.any(axis=2)
            masks.append(torch.as_tensor(mask, dtype=torch.uint8))

        if masks:
            masks = torch.stack(masks)
        else:
            masks = torch.zeros((0, h, w), dtype=torch.uint8)

        target = {"masks": tv_tensors.Mask(masks)}

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


# ============================================================
# Build transforms and dataloader
# ============================================================
def build_test_transforms(cfg):
    return build_augmentations(cfg, train=False)

def build_test_dataloader(cfg):
    ds = COCOSegmentationDataset(
        cfg["dataset"]["test"],
        transforms=build_test_transforms(cfg),
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=utils.collate_fn,
    )
    return dl


# ============================================================
# Visualization helper (image + GT + boxes + masks)
# ============================================================
def save_visual_example(image, true_mask, pred_mask, boxes, idx, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    vis_dir = os.path.join(out_dir, "examples")
    os.makedirs(vis_dir, exist_ok=True)

    img = image.detach().cpu().numpy()
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    img = img.transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Original
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # Ground Truth mask overlay
    axes[1].imshow(img)
    axes[1].imshow(true_mask, alpha=0.4, cmap="Greens")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Prediction (mask + boxes)
    axes[2].imshow(img)
    axes[2].imshow(pred_mask, alpha=0.4, cmap="Reds")
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="cyan", facecolor="none"
        )
        axes[2].add_patch(rect)
    axes[2].set_title("Prediction + Boxes")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"example_{idx:04d}.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Full evaluation
# ============================================================
def run_test_evaluation(
    model,
    test_dl,
    device,
    cfg,
    out_dir,
    visualize=False,
    num_examples=5,
):
    """
    Evaluates model on test set using:
      - Semantic metrics (Dice, IoU, Precision, Recall, F1)
      - Instance-level info (bubble count + area stats)
      - Timing (avg inference time + FPS)
      - Optional visualization (boxes + masks)
    """

    os.makedirs(out_dir, exist_ok=True)

    # Read eval settings from YAML
    eval_cfg = cfg.get("eval", {})
    score_thresh = float(eval_cfg.get("score_thresh", 0.5))
    min_area = int(eval_cfg.get("min_area", 50))
    mask_thresh = float(eval_cfg.get("mask_thresh", 0.5))
    example_seed = int(eval_cfg.get("example_seed", 123))

    random.seed(example_seed)
    np.random.seed(example_seed)
    torch.manual_seed(example_seed)

    # CSV header
    csv_path = os.path.join(out_dir, "test_semantic_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ImageIdx", "Precision", "Recall", "F1", "Dice", "IoU",
            "NumBubbles", "MeanArea", "MedianArea",
        ])

    stats = {
        "precision": [], "recall": [], "f1": [],
        "dice": [], "iou": [], "num_bubbles": [],
        "mean_area": [], "median_area": [],
        "infer_times": []
    }

    from sklearn.metrics import precision_score, recall_score, f1_score

    model.eval()

    num_batches = len(test_dl)
    example_indices = set(random.sample(range(num_batches), min(num_examples, num_batches)))

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_dl):

            image = images[0].to(device)
            image = image[:3, ...]  # ensure RGB
            true_mask = np.any(targets[0]["masks"].cpu().numpy(), axis=0).astype(np.uint8)

            # Skip images with no GT bubbles
            if true_mask.sum() == 0:
                continue

            # Timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            pred = model([image])[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()

            infer_time = t1 - t0
            stats["infer_times"].append(infer_time)

            # ------------------------------------------------------------------
            # Instance filtering by score + area
            # ------------------------------------------------------------------
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            masks_prob = pred["masks"].cpu().numpy()  # [N,1,H,W]

            keep_boxes = []
            keep_masks = []
            bubble_areas = []

            for b, s, m_prob in zip(boxes, scores, masks_prob):
                if s < score_thresh:
                    continue
                m_bin = (m_prob[0] > mask_thresh).astype(np.uint8)
                area = int(m_bin.sum())
                if area < min_area:
                    continue
                keep_boxes.append(b)
                keep_masks.append(m_bin)
                bubble_areas.append(area)

            keep_boxes = np.array(keep_boxes) if keep_boxes else np.zeros((0,4))
            keep_masks = np.array(keep_masks) if keep_masks else np.zeros_like(true_mask[None,...])
            bubble_areas = np.array(bubble_areas) if bubble_areas else np.array([])

            num_bubbles = len(keep_boxes)
            stats["num_bubbles"].append(num_bubbles)

            mean_area = float(bubble_areas.mean()) if bubble_areas.size else 0.0
            median_area = float(np.median(bubble_areas)) if bubble_areas.size else 0.0
            stats["mean_area"].append(mean_area)
            stats["median_area"].append(median_area)

            # ------------------------------------------------------------------
            # Semantic merging
            # ------------------------------------------------------------------
            pred_mask = np.any(keep_masks, axis=0).astype(np.uint8) if keep_masks.ndim == 3 else np.zeros_like(true_mask)

            # Semantic metrics
            pred_flat = pred_mask.flatten()
            true_flat = true_mask.flatten()

            precision = precision_score(true_flat, pred_flat, zero_division=0)
            recall = recall_score(true_flat, pred_flat, zero_division=0)
            f1 = f1_score(true_flat, pred_flat, zero_division=0)

            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            dice = (2 * intersection) / (pred_mask.sum() + true_mask.sum() + 1e-6)
            iou = intersection / (union + 1e-6)

            stats["precision"].append(float(precision))
            stats["recall"].append(float(recall))
            stats["f1"].append(float(f1))
            stats["dice"].append(float(dice))
            stats["iou"].append(float(iou))

            # Write per-image row
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, precision, recall, f1, dice, iou,
                    num_bubbles, mean_area, median_area,
                ])

            # Visualization
            if visualize or idx in example_indices:
                save_visual_example(image, true_mask, pred_mask, keep_boxes, idx, out_dir)

    # =======================================================
    # Aggregate results
    # =======================================================
    avg_metrics = {
        "precision": np.mean(stats["precision"]) if stats["precision"] else 0,
        "recall": np.mean(stats["recall"]) if stats["recall"] else 0,
        "f1": np.mean(stats["f1"]) if stats["f1"] else 0,
        "dice": np.mean(stats["dice"]) if stats["dice"] else 0,
        "iou": np.mean(stats["iou"]) if stats["iou"] else 0,
    }

    instance_metrics = {
        "avg_bubbles": np.mean(stats["num_bubbles"]) if stats["num_bubbles"] else 0,
        "mean_area": np.mean(stats["mean_area"]) if stats["mean_area"] else 0,
        "median_area": np.mean(stats["median_area"]) if stats["median_area"] else 0,
    }

    timing = {
        "avg_infer_time": np.mean(stats["infer_times"]) if stats["infer_times"] else 0,
        "fps": float(1 / np.mean(stats["infer_times"])) if stats["infer_times"] else 0,
    }

    # =======================================================
    # Save JSON summary
    # =======================================================
    results = EvalResults(
        semantic_metrics=avg_metrics,
        instance_metrics=instance_metrics,
        timing=timing,
        config_used=cfg.get("eval", {}),
        examples_dir=os.path.join(out_dir, "examples"),
    )

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results.to_dict(), f, indent=4)

    print("\n=== Test Metrics Summary ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v}")
    for k, v in instance_metrics.items():
        print(f"{k}: {v}")
    for k, v in timing.items():
        print(f"{k}: {v}")

    print("\nVisual examples saved in:", results.examples_dir)
    print("JSON summary saved in:", os.path.join(out_dir, "results.json"))
    print("CSV saved in:", csv_path)

    return results
