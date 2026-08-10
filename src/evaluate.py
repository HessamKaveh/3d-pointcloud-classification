import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_test_loader, get_class_names, NUM_CLASSES
from model import PointNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_pointnet.pt")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Train the model first with: python src/train.py"
        )

    test_loader = get_test_loader(batch_size=32)
    class_names = get_class_names()

    model = PointNet(num_classes=NUM_CLASSES).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Backward compatibility with a raw state_dict checkpoint.
        model.load_state_dict(checkpoint)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(device, non_blocking=True)
            logits = model(points)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    accuracy = float((all_preds == all_labels).mean())
    print(f"\nTest Accuracy: {accuracy:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        labels=np.arange(NUM_CLASSES),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\nClassification report:\n")
    print(report)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(
        os.path.join(RESULTS_DIR, "classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    with open(
        os.path.join(RESULTS_DIR, "test_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({"test_accuracy": accuracy}, f, indent=2)

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=np.arange(NUM_CLASSES),
    )

    fig = plt.figure(figsize=(14, 12))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks(np.arange(NUM_CLASSES), class_names, rotation=90, fontsize=6)
    plt.yticks(np.arange(NUM_CLASSES), class_names, fontsize=6)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"ModelNet40 confusion matrix — accuracy {accuracy:.4f}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "confusion_matrix.png"),
        dpi=180,
    )
    plt.close(fig)

    # Visualize three test point clouds with predicted labels.
    fig = plt.figure(figsize=(15, 5))
    dataset = test_loader.dataset

    for i in range(min(3, len(dataset))):
        points, true_label = dataset[i]

        with torch.no_grad():
            logits = model(points.unsqueeze(0).to(device))
            pred_label = int(logits.argmax(dim=1).item())

        points_np = points.numpy()
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(
            points_np[:, 0],
            points_np[:, 1],
            points_np[:, 2],
            s=1,
        )
        ax.set_title(
            f"True: {class_names[int(true_label)]}\n"
            f"Pred: {class_names[pred_label]}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "pointcloud_samples.png"),
        dpi=150,
    )
    plt.close(fig)

    print(f"\nSaved evaluation outputs to: {RESULTS_DIR}")


if __name__ == "__main__":
    evaluate()
