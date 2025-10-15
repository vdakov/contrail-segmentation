# %%
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_contrail_model(model_path, dataset="own", augmentation=None, num_images=4, device=None):
    """
    Evaluate a saved ContrailModel with Dice (F1), Precision, and Recall metrics,
    and visualize predictions alongside metrics.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    if dataset == "own":
        from data import own_dataset_2
        val_dataset = own_dataset_2(augmentation)[1]
    elif dataset == "google":
        from data import google_dataset
        val_dataset = google_dataset(augmentation=augmentation)[1]
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Load model
    model = ContrailModel(arch="UNet", in_channels=1, out_classes=1, loss="dice")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    dice_scores, precisions, recalls = [], [], []
    preds_all, targets_all, inputs_all = [], [], []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_dataloader):
            x, y = x.to(device), y.to(device)
            pred = torch.sigmoid(model(x))
            pred_bin = (pred > 0.5).float()

            intersection = (pred_bin * y).sum(dim=[1,2,3])
            dice = (2.0 * intersection) / (pred_bin.sum(dim=[1,2,3]) + y.sum(dim=[1,2,3]) + 1e-6)
            dice_scores.extend(dice.cpu().numpy())

            tp = intersection
            fp = (pred_bin * (1 - y)).sum(dim=[1,2,3])
            fn = ((1 - pred_bin) * y).sum(dim=[1,2,3])
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            precisions.extend(precision.cpu().numpy())
            recalls.extend(recall.cpu().numpy())

            preds_all.append(pred_bin.cpu())
            targets_all.append(y.cpu())
            inputs_all.append(x.cpu())

            if (i+1) * val_dataloader.batch_size >= num_images:
                break

    avg_dice = np.mean(dice_scores)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    print(f"Evaluation on {dataset} dataset:")
    print(f"Dice (F1): {avg_dice:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")

    # --- Visualization ---
    preds = torch.cat(preds_all)[:num_images].numpy()
    targets = torch.cat(targets_all)[:num_images].numpy()
    inputs = torch.cat(inputs_all)[:num_images].numpy()

    fig, axes = plt.subplots(num_images, 4, figsize=(14, 3*num_images))
    fig.suptitle(f"Evaluation Results - {dataset}\nPrecision={avg_precision:.3f}, Recall={avg_recall:.3f}, F1={avg_dice:.3f}",
                 fontsize=14, fontweight="bold")

    for i in range(num_images):
        axes[i, 0].imshow(inputs[i, 0], cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(targets[i, 0], cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i, 0], cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

        # Show combined overlay or per-image stats
        intersection = np.logical_and(preds[i, 0] > 0.5, targets[i, 0] > 0.5)
        union = np.logical_or(preds[i, 0] > 0.5, targets[i, 0] > 0.5)
        overlay = np.zeros((*preds[i, 0].shape, 3))
        overlay[..., 0] = targets[i, 0]  # red = ground truth
        overlay[..., 1] = preds[i, 0]    # green = prediction
        overlay[..., 2] = intersection   # yellow = overlap
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay (GT=R, Pred=G)")
        axes[i, 3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return avg_dice, avg_precision, avg_recall
