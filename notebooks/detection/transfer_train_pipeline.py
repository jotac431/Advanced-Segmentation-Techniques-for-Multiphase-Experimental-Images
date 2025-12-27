# transfer_train_pipeline.py

import os
import csv
import yaml
import inspect
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from models import MODEL_REGISTRY
from utils import *
from engine import train_one_epoch, evaluate

# Reuse your dataset + transforms + backbone freeze utilities
from synthetic_train_pipeline import (
    set_seed,
    flatten_dict,
    build_dataloaders,
    freeze_backbone,
    unfreeze_backbone,
)

def _safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _build_model_from_registry(model_cfg):
    """Call MODEL_REGISTRY safely even if builders have different signatures."""
    builder = MODEL_REGISTRY[model_cfg["type"]]
    sig = inspect.signature(builder)

    kwargs = {}
    if "pretrained_backbone" in sig.parameters:
        kwargs["pretrained_backbone"] = model_cfg.get("pretrained_backbone", True)

    # Most of your builders likely take (num_classes) positionally
    return builder(model_cfg["num_classes"], **kwargs)

def _load_checkpoint(model, ckpt_path, device, strict=True):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Your synthetic pipeline saves model.state_dict() directly
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        model.load_state_dict(ckpt, strict=strict)
    else:
        # Fallback if you ever save {"model": state_dict, ...}
        model.load_state_dict(ckpt["model"], strict=strict)

def _make_optimizer(model, cfg):
    lr = cfg["train"]["lr"]
    backbone_factor = cfg["train"]["backbone_lr_factor"]

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    return torch.optim.SGD(
        [
            {"params": head_params, "lr": lr},
            {"params": backbone_params, "lr": lr * backbone_factor},
        ],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )

def run_transfer_training(cfg):
    # -------- seed (compatible with your current YAML) --------
    seed = _safe_get(cfg, ["experiment", "seed"], 42)
    set_seed(seed)

    device, train_dl, val_dl, test_dl = build_dataloaders(cfg)

    # -------- model --------
    model = _build_model_from_registry(cfg["model"])
    model.to(device)

    # -------- load synthetic weights --------
    ckpt_path = _safe_get(cfg, ["transfer", "init_checkpoint"], None)
    if ckpt_path:
        strict = _safe_get(cfg, ["transfer", "strict_load"], True)
        _load_checkpoint(model, ckpt_path, device=device, strict=strict)
        print(f"✅ Loaded init checkpoint: {ckpt_path}")
    else:
        print("⚠️ No transfer.init_checkpoint provided; training from current init.")

    # -------- logging --------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = cfg["experiment"]["name"]
    out_dir = os.path.join(cfg["output"]["dir"], f"{exp_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    use_tb = cfg["logging"].get("use_tensorboard", True)
    writer = None
    if use_tb:
        writer = SummaryWriter(log_dir=os.path.join(cfg["logging"]["log_dir"], f"{exp_name}_{timestamp}"))
        writer.add_text("config", f"```yaml\n{yaml.dump(cfg, sort_keys=False)}\n```")

    # -------- training controls --------
    num_epochs = cfg["train"]["epochs"]
    patience = cfg["train"].get("patience", 5)

    freeze_epochs = _safe_get(cfg, ["transfer", "freeze_backbone_epochs"], 0)
    best_val = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    # CSV
    csv_path = os.path.join(out_dir, f"{exp_name}_metrics.csv")

    # AMP
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    # -------- phase 1: optional backbone freeze --------
    if freeze_epochs > 0:
        freeze_backbone(model)

    optimizer = _make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["train"]["lr_step_size"],
        gamma=cfg["train"]["gamma"],
    )

    print(f"\n======== Transfer Learning: {exp_name} ========\n")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("LRs:", [g["lr"] for g in optimizer.param_groups])

        # unfreeze at boundary and rebuild optimizer (so backbone params enter training cleanly)
        if freeze_epochs > 0 and epoch == freeze_epochs:
            unfreeze_backbone(model)
            optimizer = _make_optimizer(model, cfg)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg["train"]["lr_step_size"],
                gamma=cfg["train"]["gamma"],
            )

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

        if writer:
            for name, meter in metric_logger.meters.items():
                writer.add_scalar(f"Train/{name}", meter.global_avg, epoch)

        # ---- validation (Mask R-CNN COCO metrics) ----
        print("Validating.")
        coco_eval = evaluate(model, val_dl, device=device)
        segm_stats = coco_eval.coco_eval["segm"].stats

        current_val = float(segm_stats[0])  # AP@[.50:.95]

        if writer:
            writer.add_scalar("Val/mAP", segm_stats[0], epoch)
            writer.add_scalar("Val/mAP50", segm_stats[1], epoch)
            writer.add_scalar("Val/mAP75", segm_stats[2], epoch)

        # CSV write
        write_header = (not os.path.exists(csv_path))
        with open(csv_path, mode="a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "epoch",
                    "AP@[0.50:0.95]", "AP@0.50", "AP@0.75",
                    "AP_small", "AP_medium", "AP_large",
                    "AR@1", "AR@10", "AR@100",
                    "AR_small", "AR_medium", "AR_large",
                ])
            w.writerow([epoch] + list(segm_stats))

        # ---- early stopping + checkpoint ----
        if current_val > best_val:
            print(f"New best mAP: {current_val:.4f} (prev {best_val:.4f})")
            best_val = current_val
            best_epoch = epoch + 1
            epochs_no_improve = 0

            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pth"))
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if writer:
        writer.add_hparams(flatten_dict(cfg), {"best_val_metric": best_val})
        writer.close()

    print(f"Done. Best epoch={best_epoch}, best mAP={best_val:.4f}")
    return model, best_val, best_epoch, out_dir
