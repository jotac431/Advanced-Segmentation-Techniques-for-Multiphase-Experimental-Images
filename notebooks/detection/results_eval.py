from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def evaluate_segmentation_metrics(model, data_loader, device, max_images=None, visualize=False):
    model.eval()
    dice_scores, iou_scores = [], []
    precisions, recalls, f1_scores = [], [], []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            if max_images is not None and idx >= max_images:
                break

            image = images[0].to(device)
            image = image[:3, ...]  # remove alpha if present
            target = targets[0]

            prediction = model([image])[0]

            pred_masks = (prediction["masks"].cpu() > 0.7).squeeze(1).numpy()
            true_masks = target["masks"].cpu().numpy()

            pred_mask = np.any(pred_masks, axis=0).astype(np.uint8)
            true_mask = np.any(true_masks, axis=0).astype(np.uint8)

            if true_mask.sum() == 0:
                continue

            pred_flat = pred_mask.flatten()
            true_flat = true_mask.flatten()

            # Compute metrics
            precision = precision_score(true_flat, pred_flat, zero_division=0)
            recall = recall_score(true_flat, pred_flat, zero_division=0)
            f1 = f1_score(true_flat, pred_flat, zero_division=0)

            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            dice = (2 * intersection) / (pred_mask.sum() + true_mask.sum() + 1e-6)
            iou = intersection / (union + 1e-6)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            dice_scores.append(dice)
            iou_scores.append(iou)

            # Optional visualization
            if visualize:
                x_cpu = image.cpu()
                vis_img = (255.0 * (x_cpu - x_cpu.min()) / (x_cpu.max() - x_cpu.min())).to(torch.uint8)
                pred_labels = [f"bubble: {score:.2f}" for score in prediction["scores"]]
                pred_boxes = prediction["boxes"].cpu().long()
                masks = (prediction["masks"].cpu() > 0.7).squeeze(1)
                output_image = draw_bounding_boxes(vis_img, pred_boxes, pred_labels, colors="red")
                output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

                plt.figure(figsize=(10, 10))
                plt.imshow(output_image.permute(1, 2, 0))
                plt.axis("off")
                plt.title(f"Sample #{idx}")
                plt.show()

    print("\nFinal Evaluation Metrics:")
    print(f"Average Precision:         {np.mean(precisions):.4f}")
    print(f"Average Recall:            {np.mean(recalls):.4f}")
    print(f"Average F1-score:          {np.mean(f1_scores):.4f}")
    print(f"Average Dice Coefficient:  {np.mean(dice_scores):.4f}")
    print(f"Average IoU (Jaccard):     {np.mean(iou_scores):.4f}")
