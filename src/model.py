"""
src/model.py — PyTorch Model Definitions
==========================================
Two model architectures for binary speech sentiment classification:

Option A — FCClassifier (Fully Connected baseline):
  Flattens the MFCC matrix and passes through 3 FC layers with
  BatchNorm and Dropout. Fast to train; good for quick validation.
  Architecture: Flatten → FC(4096) → FC(512) → FC(64) → FC(2)

Option B — CNNClassifier (1D Convolutional Network, recommended):
  Treats each MFCC coefficient row as a channel and applies 1D
  convolutions along the time axis. Better at capturing temporal
  patterns in speech. Uses global average pooling to handle variable
  lengths gracefully.
  Architecture: 3× [Conv1D → BN → ReLU → MaxPool] → GAP → FC(2)

Both models:
  - Use Dropout (0.3–0.5) to prevent overfitting on small datasets
  - Output 2 logits (Negative, Positive) — use softmax for probabilities
  - Share the same forward() signature for transparent swap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.features import N_MFCC, MAX_LEN_FRAMES

# Input feature shape: (batch, channels, time)
# channels = 3 * N_MFCC = 120  (mfcc + delta + delta-delta)
# time     = MAX_LEN_FRAMES = 200
N_FEATURES = 3 * N_MFCC   # 120 feature rows
N_CLASSES  = 2             # Positive, Negative


# ─── Option A: Fully Connected Classifier (Baseline) ─────────────────────────

class FCClassifier(nn.Module):
    """
    Fully-connected baseline classifier.

    Flattens the 2D MFCC feature matrix into a 1D vector, then
    applies three fully-connected layers with BatchNorm and Dropout.

    Input:  (batch, N_FEATURES, MAX_LEN_FRAMES) = (B, 120, 200)
    Output: (batch, 2) — raw logits for [Negative, Positive]
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        in_dim = N_FEATURES * MAX_LEN_FRAMES  # 120 * 200 = 24,000

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Layer 1: high-dim → 512
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Layer 2: 512 → 128
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Layer 3: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            # Output: 64 → 2
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 120, 200) → logits: (batch, 2)"""
        return self.classifier(x)


# ─── Shared Conv Block ────────────────────────────────────────────────────────

class ConvBlock1D(nn.Module):
    """
    A single 1D Convolutional block:
      Conv1d(in_ch → out_ch, kernel) → BatchNorm → ReLU → MaxPool

    Applied along the time axis of the MFCC feature matrix.
    Each block doubles the number of feature maps and halves temporal resolution.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5, pool_size: int = 2, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,   # same padding to preserve time dim
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Option B: 1D CNN Classifier (Recommended) ───────────────────────────────

class CNNClassifier(nn.Module):
    """
    1D Convolutional classifier for speech sentiment.

    Treats the 120 MFCC feature rows as channels and applies three
    Conv1D blocks along the time axis (200 frames), progressively
    reducing temporal resolution while increasing depth.

    Global Average Pooling (GAP) at the end removes the time dimension —
    this also provides mild regularization and handles variable-length input.

    Architecture:
      Input (B, 120, 200)
        → ConvBlock(120 → 256, k=5) → (B, 256, 100)
        → ConvBlock(256 → 128, k=5) → (B, 128,  50)
        → ConvBlock(128 →  64, k=3) → (B,  64,  25)
        → GlobalAvgPool             → (B,  64)
        → Dropout(0.5)
        → Linear(64 → 2)            → (B,   2)

    Input:  (batch, N_FEATURES, MAX_LEN_FRAMES) = (B, 120, 200)
    Output: (batch, 2) — raw logits for [Negative, Positive]
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        # Three conv blocks: progressively compress temporal resolution
        self.conv1 = ConvBlock1D(N_FEATURES, 256, kernel_size=5, pool_size=2, dropout=0.3)
        self.conv2 = ConvBlock1D(256,        128, kernel_size=5, pool_size=2, dropout=0.3)
        self.conv3 = ConvBlock1D(128,         64, kernel_size=3, pool_size=2, dropout=0.3)

        # Global Average Pooling: average over remaining time frames → (B, 64)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 120, 200) → logits: (batch, 2)"""
        x = self.conv1(x)   # (B, 256, 100)
        x = self.conv2(x)   # (B, 128,  50)
        x = self.conv3(x)   # (B,  64,  25)
        x = self.gap(x)     # (B,  64,   1)
        x = x.squeeze(-1)   # (B,  64)
        return self.head(x) # (B,   2)


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(model_type: str = "cnn", dropout: float = 0.5) -> nn.Module:
    """
    Instantiate and return the requested model architecture.

    Parameters
    ----------
    model_type : str
        'fc'  → FCClassifier (baseline, faster to train)
        'cnn' → CNNClassifier (recommended, higher accuracy)
    dropout : float
        Dropout rate to apply across all layers.

    Returns
    -------
    nn.Module ready for training.
    """
    model_type = model_type.lower()
    if model_type == "fc":
        model = FCClassifier(dropout=dropout)
    elif model_type == "cnn":
        model = CNNClassifier(dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type='{model_type}'. Choose 'fc' or 'cnn'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model.__class__.__name__} | Parameters: {n_params:,}")
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
