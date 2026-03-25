"""
Evaluation script for the EdgeHAR CNN-LSTM model.

Loads the best saved model checkpoint and evaluates it on the test set.
Generates:
    - Full classification report (sklearn)
    - Confusion matrix heatmap (seaborn)
    - Per-class F1 scores, macro F1, test accuracy
    - Inference time per sample

Usage:
    python src/evaluate.py
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from dataset import create_dataloaders
from model import CNNLSTM

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG = {
    "project_root": Path(__file__).resolve().parent.parent,
    "model_path": Path(__file__).resolve().parent.parent / "models" / "best_model.pth",
    "output_dir": Path(__file__).resolve().parent.parent / "outputs",
    "confusion_matrix_name": "confusion_matrix.png",
    "seed": 42,
}


def load_model(model_path: Path, device: torch.device) -> tuple:
    """Load the best saved model checkpoint.

    Args:
        model_path: Path to the saved model checkpoint.
        device: Device to load the model onto.

    Returns:
        Tuple of (model, checkpoint_dict).

    Raises:
        FileNotFoundError: If model checkpoint is not found.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ Model checkpoint not found at: {model_path}\n"
            f"   Please run training first: python src/train.py"
        )

    print(f"📥 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", {})
    model = CNNLSTM(
        num_channels=model_config.get("num_channels", 6),
        num_classes=model_config.get("num_classes", 6),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"   Loaded from epoch {checkpoint.get('epoch', '?')} "
          f"(val_acc: {checkpoint.get('val_acc', 0):.2%})")

    return model, checkpoint


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """Run evaluation on the test set.

    Args:
        model: Trained model in eval mode.
        test_loader: Test data loader.
        device: Device to run inference on.

    Returns:
        Tuple of (all_preds, all_labels, avg_inference_time_ms).
    """
    all_preds = []
    all_labels = []
    inference_times = []

    for signals, labels in test_loader:
        signals = signals.to(device)

        start = time.perf_counter()
        outputs = model(signals)
        end = time.perf_counter()

        inference_times.append((end - start) / signals.size(0))

        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_time_ms = np.mean(inference_times) * 1000

    return all_preds, all_labels, avg_time_ms


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Path,
) -> None:
    """Generate and save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.
        save_path: Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        square=True,
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("EdgeHAR — Confusion Matrix", fontsize=15, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved to: {save_path}")


def main() -> None:
    """Main evaluation function."""
    # Setup
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}\n")

    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    # Load model
    model, checkpoint = load_model(CONFIG["model_path"], device)
    class_names = checkpoint.get("class_names", [
        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
        "SITTING", "STANDING", "LAYING"
    ])

    # Load test data
    _, test_loader, _ = create_dataloaders(batch_size=64)

    # Evaluate
    print("\n🧪 Running evaluation on test set...\n")
    y_pred, y_true, avg_inference_ms = evaluate_model(model, test_loader, device)

    # ─── Results ─────────────────────────────────────────────────────────
    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    print("=" * 70)
    print("📋 Classification Report")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("=" * 70)
    print("📊 Summary Metrics")
    print("=" * 70)
    print(f"   Test Accuracy     : {test_acc:.4f} ({test_acc:.2%})")
    print(f"   Macro F1 Score    : {macro_f1:.4f}")
    print(f"   Inference Time    : {avg_inference_ms:.3f} ms/sample")
    print()

    print("   Per-Class F1 Scores:")
    for name, f1_val in zip(class_names, per_class_f1):
        bar = "█" * int(f1_val * 30)
        print(f"     {name:25s} : {f1_val:.4f}  {bar}")

    print("=" * 70)

    # Confusion matrix plot
    cm_path = CONFIG["output_dir"] / CONFIG["confusion_matrix_name"]
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
