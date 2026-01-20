from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import os
import csv
import time
from scipy.ndimage import center_of_mass
from PIL import ImageFont, ImageDraw, Image
import torchvision.transforms.functional as F

# Make this clear in your report:
# "Mask R-CNN, an instance segmentation model, was evaluated by merging all predicted masks into a single binary mask to enable direct comparison with UNet in a pixel-wise semantic segmentation context.”
def evaluate_segmentation_metrics(model, data_loader, device, model_name="model", max_images=None, visualize=False):
    model.eval()
    dice_scores, iou_scores = [], []
    precisions, recalls, f1_scores = [], [], []
    infer_times = []

    # Setup CSV saving (only if not visualizing)
    save_path = f"results/{model_name}/per_image_segmentation_metrics.csv"

    if not visualize:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ImageIdx", "Precision", "Recall", "F1", "Dice", "IoU"])

    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            if max_images is not None and idx >= max_images:
                break

            image = images[0].to(device)
            image = image[:3, ...]  # remove alpha if present
            target = targets[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            prediction = model([image])[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            infer_times.append(end_time - start_time)

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

            # Save per-image metrics
            if not visualize:
                with open(save_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([idx, precision, recall, f1, dice, iou])

            # Optional visualization
            if visualize:
                x_cpu = image.cpu()
                vis_img = (255.0 * (x_cpu - x_cpu.min()) / (x_cpu.max() - x_cpu.min())).to(torch.uint8)
                pred_labels = [f"bubble: {score:.2f}" for score in prediction["scores"]]
                pred_boxes = prediction["boxes"].cpu().long()
                masks = (prediction["masks"].cpu() > 0.7).squeeze(1)
                output_image = draw_bounding_boxes(vis_img, pred_boxes, pred_labels, colors="red")
                output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

                # Prepare overlay text
                bubble_count = len(prediction["scores"])
                metrics_text = (
                    f"Image #{idx} | Bubbles: {bubble_count}\n"
                    f"P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}\n"
                    f"Dice: {dice:.2f}, IoU: {iou:.2f}"
                )

                # Draw text on image
                # Convert to PIL for annotation
                img_pil = torch.permute(output_image, (1, 2, 0)).numpy()
                img_pil = Image.fromarray(img_pil)

                # Optional: load a TTF font (if desired)
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 20)
                except:
                    font = ImageFont.load_default()

                draw = ImageDraw.Draw(img_pil)
                draw.rectangle([5, 5, 400, 70], fill=(255, 255, 255, 200))  # background box
                draw.text((10, 10), metrics_text, fill="black", font=font)

                # Display
                plt.figure(figsize=(10, 10))
                plt.imshow(img_pil)
                plt.axis("off")
                plt.title(f"Sample #{idx}")
                plt.show()

    # Final averages
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print("\nFinal Evaluation Metrics:")
    print(f"Average Precision:         {avg_precision:.4f}")
    print(f"Average Recall:            {avg_recall:.4f}")
    print(f"Average F1-score:          {avg_f1:.4f}")
    print(f"Average Dice Coefficient:  {avg_dice:.4f}")
    print(f"Average IoU (Jaccard):     {avg_iou:.4f}")
    
    if len(infer_times) > 0:
        avg_time = np.mean(infer_times)
        std_time = np.std(infer_times)
        max_time = np.max(infer_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        print("\n--- Inference Timing Stats ---")
        print(f"Average inference time per image: {avg_time*1000:.2f} ms")
        print(f"Standard deviation:               {std_time*1000:.2f} ms")
        print(f"Maximum inference time:          {max_time*1000:.2f} ms")
        print(f"Estimated FPS:                    {fps:.2f}")


    # Save average metrics to CSV
    if not visualize:
        with open(save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["AVERAGES", avg_precision, avg_recall, avg_f1, avg_dice, avg_iou])

        # Save timing stats to separate TXT file
        timing_path = f"results/{model_name}/timing_stats.txt"
        with open(timing_path, "w") as f:
            f.write("Inference Timing Statistics:\n")
            f.write(f"Average inference time per image: {avg_time*1000:.2f} ms\n")
            f.write(f"Standard deviation:               {std_time*1000:.2f} ms\n")
            f.write(f"Maximum inference time:          {max_time*1000:.2f} ms\n")
            f.write(f"Estimated FPS:                    {fps:.2f}\n")


def evaluate_unet_segmentation_metrics(model, data_loader, device, model_name="unet", max_images=None, visualize=False):

    model.eval()
    dice_scores, iou_scores = [], []
    precisions, recalls, f1_scores = [], [], []
    infer_times = []

    save_path = f"results/{model_name}/per_image_segmentation_metrics.csv"

    if not visualize:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ImageIdx", "Precision", "Recall", "F1", "Dice", "IoU"])

    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            if max_images is not None and idx >= max_images:
                break

            image = images[0].to(device)[:3, ...]  # Remove alpha if present
            target = targets[0]
            true_mask = np.any(target["masks"].cpu().numpy(), axis=0).astype(np.uint8)

            # Inference timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            predictions = model([image])

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            pred = predictions[0]

            infer_times.append(end_time - start_time)

            # Apply filtering like in your working code
            score_thresh = 0
            min_area = 0

            scores = pred["scores"].cpu()
            keep = scores > score_thresh
            masks = pred["masks"].cpu()[keep]
            scores = scores[keep]

            # Area filter
            areas = masks.sum(dim=[1, 2])
            keep_area = areas > min_area
            masks = (masks[keep_area] > 0.5).numpy().astype(np.uint8)

            # Combine masks
            if len(masks) > 0:
                pred_mask = np.any(masks, axis=0).astype(np.uint8)
            else:
                pred_mask = np.zeros_like(true_mask)

            # Compute metrics as before
            pred_flat = pred_mask.flatten()
            true_flat = true_mask.flatten()

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

            # Save per-image metrics
            if not visualize:
                with open(save_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([idx, precision, recall, f1, dice, iou])

            # Optional visualization
            if visualize:
                vis_img = (255.0 * (image.cpu() - image.cpu().min()) / (image.cpu().max() - image.cpu().min())).to(torch.uint8)
                vis_img = F.to_pil_image(vis_img)

                plt.figure(figsize=(10, 10))
                plt.imshow(vis_img)
                plt.imshow(true_mask, alpha=0.3, cmap="Greens")
                plt.imshow(pred_mask, alpha=0.3, cmap="Reds")
                plt.axis("off")
                plt.title(f"Image {idx}: Green=GT | Red=Pred")
                plt.show()

    # Summary metrics
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print("\nFinal Evaluation Metrics:")
    print(f"Average Precision:         {avg_precision:.4f}")
    print(f"Average Recall:            {avg_recall:.4f}")
    print(f"Average F1-score:          {avg_f1:.4f}")
    print(f"Average Dice Coefficient:  {avg_dice:.4f}")
    print(f"Average IoU (Jaccard):     {avg_iou:.4f}")

    if infer_times:
        avg_time = np.mean(infer_times)
        std_time = np.std(infer_times)
        max_time = np.max(infer_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        print("\n--- Inference Timing Stats ---")
        print(f"Average inference time per image: {avg_time*1000:.2f} ms")
        print(f"Standard deviation:               {std_time*1000:.2f} ms")
        print(f"Maximum inference time:          {max_time*1000:.2f} ms")
        print(f"Estimated FPS:                    {fps:.2f}")

        if not visualize:
            timing_path = f"results/{model_name}/timing_stats.txt"
            with open(timing_path, "w") as f:
                f.write("Inference Timing Statistics:\n")
                f.write(f"Average inference time per image: {avg_time*1000:.2f} ms\n")
                f.write(f"Standard deviation:               {std_time*1000:.2f} ms\n")
                f.write(f"Maximum inference time:          {max_time*1000:.2f} ms\n")
                f.write(f"Estimated FPS:                    {fps:.2f}\n")

            with open(save_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(["AVERAGES", avg_precision, avg_recall, avg_f1, avg_dice, avg_iou])

def extract_tensor(output):
    # Recursively dig into dicts/lists/tuples to get to the tensor
    while isinstance(output, (dict, list, tuple)):
        if isinstance(output, dict):
            if "out" in output:
                output = output["out"]
            else:
                output = list(output.values())[0]
        elif isinstance(output, (list, tuple)):
            output = output[0]
    return output
