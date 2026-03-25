"""
Training script for the EdgeHAR CNN-LSTM model.

Implements a full training loop with:
    - AdamW optimizer with CosineAnnealingLR scheduler
    - Early stopping on validation loss
    - Model checkpointing (best validation loss)
    - Training history logging and curve plotting

Usage:
    python src/train.py --epochs 50 --lr 0.001 --batch_size 64 --patience 10
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_dataloaders
from model import CNNLSTM

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG = {
    "project_root": Path(__file__).resolve().parent.parent,
    "seed": 42,
    "model_save_dir": Path(__file__).resolve().parent.parent / "models",
    "output_dir": Path(__file__).resolve().parent.parent / "outputs",
    "best_model_name": "best_model.pth",
    "history_name": "training_history.json",
    "curves_name": "training_curves.png",
}


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value. Defaults to 42.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: The neural network model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * signals.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model on the test/validation set.

    Args:
        model: The neural network model.
        loader: Validation data loader.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Tuple of (average_loss, accuracy) for the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        outputs = model(signals)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * signals.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_training_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    """Plot and save training/validation loss and accuracy curves.

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss',
                 'train_acc', 'val_acc'.
        save_path: Path to save the plot image.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy subplot
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Training curves saved to: {save_path}")


def main() -> None:
    """Main training function."""
    # ─── Parse Arguments ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train the EdgeHAR CNN-LSTM model"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (default: 10)"
    )
    args = parser.parse_args()

    # ─── Setup ───────────────────────────────────────────────────────────
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    print(f"   Epochs: {args.epochs} | LR: {args.lr} | "
          f"Batch: {args.batch_size} | Patience: {args.patience}\n")

    # Create output directories
    CONFIG["model_save_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    # ─── Data ────────────────────────────────────────────────────────────
    train_loader, test_loader, class_names = create_dataloaders(
        batch_size=args.batch_size,
    )

    # ─── Model ───────────────────────────────────────────────────────────
    model = CNNLSTM(num_channels=6, num_classes=len(class_names)).to(device)
    print(model.summary())

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ─── Training Loop ───────────────────────────────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = CONFIG["model_save_dir"] / CONFIG["best_model_name"]

    print("\n" + "=" * 70)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>10}")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Print epoch results
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            marker = " ★ best"

            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "model_config": {
                        "num_channels": 6,
                        "num_classes": len(class_names),
                    },
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
            f"{val_loss:>8.4f} | {val_acc:>6.2%} | {current_lr:>10.6f}{marker}"
        )

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    total_time = time.time() - start_time

    # ─── Save Results ────────────────────────────────────────────────────
    # Save training history
    history_path = CONFIG["output_dir"] / CONFIG["history_name"]
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 Training history saved to: {history_path}")

    # Plot training curves
    curves_path = CONFIG["output_dir"] / CONFIG["curves_name"]
    plot_training_curves(history, curves_path)

    # ─── Final Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🏁 Training Complete!")
    print("=" * 70)
    print(f"   Total training time  : {total_time:.1f}s")
    print(f"   Best validation loss : {best_val_loss:.4f}")
    print(f"   Best validation acc  : {best_val_acc:.2%}")
    print(f"   Best model saved to  : {best_model_path}")
    print(f"   Training curves      : {curves_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
