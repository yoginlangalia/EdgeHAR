"""
PyTorch Dataset and DataLoader for the UCI HAR Dataset.

Loads raw inertial signal data (accelerometer + gyroscope) from the UCI
Human Activity Recognition dataset. Each sample consists of 128 timesteps
across 6 sensor channels.

Channels:
    - total_acc_x, total_acc_y, total_acc_z (accelerometer)
    - body_gyro_x, body_gyro_y, body_gyro_z (gyroscope)

Labels (0-indexed):
    0: WALKING, 1: WALKING_UPSTAIRS, 2: WALKING_DOWNSTAIRS,
    3: SITTING, 4: STANDING, 5: LAYING
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG = {
    "data_root": Path(__file__).resolve().parent.parent / "data",
    "seed": 42,
    "signal_files": [
        "total_acc_x_{}.txt",
        "total_acc_y_{}.txt",
        "total_acc_z_{}.txt",
        "body_gyro_x_{}.txt",
        "body_gyro_y_{}.txt",
        "body_gyro_z_{}.txt",
    ],
    "class_names": [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ],
    "num_channels": 6,
    "sequence_length": 128,
}


def _find_dataset_dir(data_root: Path) -> Path:
    """Locate the UCI HAR Dataset directory.

    The UCI zip may extract to 'UCI HAR Dataset' (with spaces) or
    'UCI_HAR_Dataset' (with underscores). This function checks both.

    Args:
        data_root: The root data directory to search in.

    Returns:
        Path to the UCI HAR Dataset directory.

    Raises:
        FileNotFoundError: If the dataset directory is not found.
    """
    possible_names = ["UCI HAR Dataset", "UCI_HAR_Dataset"]
    for name in possible_names:
        candidate = data_root / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"UCI HAR Dataset not found in {data_root}. "
        f"Please run 'python data/download_data.py' first.\n"
        f"Expected one of: {possible_names}"
    )


def _load_signals(dataset_dir: Path, split: str) -> np.ndarray:
    """Load raw inertial signal data for a given split.

    Args:
        dataset_dir: Path to the UCI HAR Dataset root directory.
        split: Either 'train' or 'test'.

    Returns:
        Numpy array of shape (num_samples, num_channels, sequence_length).

    Raises:
        FileNotFoundError: If signal files are not found.
    """
    signals = []
    inertial_dir = dataset_dir / split / "Inertial Signals"

    if not inertial_dir.exists():
        raise FileNotFoundError(
            f"Inertial Signals directory not found at {inertial_dir}. "
            f"Please ensure the dataset is properly extracted."
        )

    for signal_template in CONFIG["signal_files"]:
        filename = signal_template.format(split)
        filepath = inertial_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Signal file not found: {filepath}. "
                f"Dataset may be incomplete."
            )

        # Each file has shape (num_samples, 128) — space-separated values
        data = np.loadtxt(filepath)
        signals.append(data)

    # Stack to shape (num_samples, num_channels, sequence_length)
    return np.stack(signals, axis=1).astype(np.float32)


def _load_labels(dataset_dir: Path, split: str) -> np.ndarray:
    """Load activity labels for a given split.

    UCI HAR labels are 1-indexed (1-6), we convert to 0-indexed (0-5).

    Args:
        dataset_dir: Path to the UCI HAR Dataset root directory.
        split: Either 'train' or 'test'.

    Returns:
        Numpy array of integer labels, shape (num_samples,), 0-indexed.

    Raises:
        FileNotFoundError: If label file is not found.
    """
    label_file = dataset_dir / split / f"y_{split}.txt"

    if not label_file.exists():
        raise FileNotFoundError(
            f"Label file not found: {label_file}. "
            f"Please ensure the dataset is properly extracted."
        )

    # Convert from 1-indexed to 0-indexed
    labels = np.loadtxt(label_file, dtype=np.int64) - 1
    return labels


class HARDataset(Dataset):
    """PyTorch Dataset for UCI Human Activity Recognition data.

    Each item returns a tuple of (signal_tensor, label) where:
        - signal_tensor: FloatTensor of shape (6, 128)
          [channels: total_acc_x/y/z, body_gyro_x/y/z]
        - label: Integer label (0-5) for the activity class

    Args:
        split: Either 'train' or 'test'.
        data_root: Path to the data directory. Defaults to CONFIG setting.

    Example:
        >>> dataset = HARDataset(split='train')
        >>> signal, label = dataset[0]
        >>> signal.shape
        torch.Size([6, 128])
    """

    def __init__(self, split: str, data_root: Path | None = None) -> None:
        super().__init__()
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.split = split
        data_root = data_root or CONFIG["data_root"]
        dataset_dir = _find_dataset_dir(data_root)

        print(f"Loading {split} data from {dataset_dir}...")
        self.signals = _load_signals(dataset_dir, split)
        self.labels = _load_labels(dataset_dir, split)

        assert len(self.signals) == len(self.labels), (
            f"Mismatch: {len(self.signals)} signals vs {len(self.labels)} labels"
        )

        print(
            f"  Loaded {len(self)} samples | "
            f"Signal shape: {self.signals.shape} | "
            f"Labels: {np.unique(self.labels)}"
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (signal_tensor, label) where signal_tensor has
            shape (6, 128) and label is an integer in [0, 5].
        """
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        label = int(self.labels[idx])
        return signal, label


def create_dataloaders(
    batch_size: int = 64,
    data_root: Path | None = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train and test DataLoaders for the UCI HAR Dataset.

    Args:
        batch_size: Batch size for both loaders. Defaults to 64.
        data_root: Path to the data directory. Defaults to CONFIG setting.
        num_workers: Number of workers for data loading. Defaults to 0.

    Returns:
        Tuple of (train_loader, test_loader, class_names) where:
            - train_loader: DataLoader for training data
            - test_loader: DataLoader for test data
            - class_names: List of 6 activity class names
    """
    # Set seed for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    train_dataset = HARDataset(split="train", data_root=data_root)
    test_dataset = HARDataset(split="test", data_root=data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"\n📊 DataLoaders created:"
        f"\n   Train: {len(train_dataset)} samples, {len(train_loader)} batches"
        f"\n   Test:  {len(test_dataset)} samples, {len(test_loader)} batches"
        f"\n   Batch size: {batch_size}"
    )

    return train_loader, test_loader, CONFIG["class_names"]
