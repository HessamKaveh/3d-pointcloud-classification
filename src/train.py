import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders, NUM_CLASSES
from model import PointNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for points, labels in loader:
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(points)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return running_loss / total, correct / total


def save_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Classification accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)


def train():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        val_fraction=0.1,
        seed=SEED,
    )

    print(f"Training samples:   {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = PointNet(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_acc = -1.0
    best_path = os.path.join(RESULTS_DIR, "best_pointnet.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "num_points": 1024,
                    "best_val_acc": best_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    with open(os.path.join(RESULTS_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_training_curves(history)

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print("Run evaluation with: python src/evaluate.py")


if __name__ == "__main__":
    train()
