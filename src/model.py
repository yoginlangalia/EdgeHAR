"""
CNN-LSTM Model for Human Activity Recognition.

Architecture:
    Input (batch, 6, 128)
    → Conv Block 1: Conv1d(6→64)  → BN → ReLU → MaxPool(2)   → (batch, 64, 64)
    → Conv Block 2: Conv1d(64→128) → BN → ReLU → MaxPool(2)  → (batch, 128, 32)
    → Conv Block 3: Conv1d(128→256) → BN → ReLU               → (batch, 256, 32)
    → Reshape to (batch, 32, 256) for LSTM
    → LSTM(256→128, 2 layers, dropout=0.3)
    → Last hidden state → (batch, 128)
    → FC: Linear(128→64) → ReLU → Dropout(0.5) → Linear(64→6)
    Output: (batch, 6) — logits for 6 activity classes
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """CNN-LSTM model for time-series activity recognition.

    Combines convolutional feature extraction with LSTM temporal modeling
    for classifying human activities from 6-channel sensor data.

    Args:
        num_channels: Number of input sensor channels. Defaults to 6.
        num_classes: Number of activity classes. Defaults to 6.
        lstm_hidden: LSTM hidden dimension. Defaults to 128.
        lstm_layers: Number of LSTM layers. Defaults to 2.
        lstm_dropout: Dropout between LSTM layers. Defaults to 0.3.
        fc_dropout: Dropout before final classification. Defaults to 0.5.

    Example:
        >>> model = CNNLSTM()
        >>> x = torch.randn(32, 6, 128)  # batch of 32
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 6])
    """

    def __init__(
        self,
        num_channels: int = 6,
        num_classes: int = 6,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        # ─── Convolutional Feature Extractor ─────────────────────────────
        # Conv Block 1: (batch, 6, 128) → (batch, 64, 64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Conv Block 2: (batch, 64, 64) → (batch, 128, 32)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Conv Block 3: (batch, 128, 32) → (batch, 256, 32)
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # ─── LSTM Temporal Encoder ───────────────────────────────────────
        # Input: (batch, seq_len=32, features=256)
        # Output: (batch, seq_len=32, hidden=128)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # ─── Classifier Head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN-LSTM model.

        Args:
            x: Input tensor of shape (batch, num_channels, sequence_length).
               Expected shape: (batch, 6, 128).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # CNN feature extraction
        x = self.conv1(x)  # (batch, 64, 64)
        x = self.conv2(x)  # (batch, 128, 32)
        x = self.conv3(x)  # (batch, 256, 32)

        # Reshape for LSTM: (batch, channels, seq) → (batch, seq, channels)
        x = x.permute(0, 2, 1)  # (batch, 32, 256)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, 32, 128)

        # Take the last hidden state
        x = lstm_out[:, -1, :]  # (batch, 128)

        # Classification
        x = self.classifier(x)  # (batch, num_classes)
        return x

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Generate a text summary of the model architecture.

        Returns:
            String containing model architecture details.
        """
        lines = [
            "=" * 60,
            "CNNLSTM Model Summary",
            "=" * 60,
            f"Input shape:  (batch, {self.num_channels}, 128)",
            f"Output shape: (batch, {self.num_classes})",
            f"Total trainable parameters: {self.count_parameters():,}",
            "-" * 60,
        ]

        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters())
            lines.append(f"  {name:20s} → {params:>10,} params")

        lines.append("=" * 60)
        return "\n".join(lines)


def build_model(
    num_channels: int = 6,
    num_classes: int = 6,
    device: str | torch.device = "cpu",
) -> CNNLSTM:
    """Build and initialize the CNN-LSTM model.

    Args:
        num_channels: Number of input sensor channels. Defaults to 6.
        num_classes: Number of activity classes. Defaults to 6.
        device: Device to place the model on. Defaults to 'cpu'.

    Returns:
        Initialized CNNLSTM model on the specified device.
    """
    model = CNNLSTM(
        num_channels=num_channels,
        num_classes=num_classes,
    )
    model = model.to(device)
    print(model.summary())
    return model
