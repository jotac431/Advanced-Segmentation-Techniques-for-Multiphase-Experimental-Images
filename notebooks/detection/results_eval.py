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


