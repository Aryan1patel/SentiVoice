"""
src/train.py — Training Loop
==============================
Trains a binary speech sentiment classifier using:
  - AdamW optimizer (weight decay for L2 regularization)
  - CosineAnnealingLR scheduler (smooth LR decay)
  - Class-weighted CrossEntropy loss (handles label imbalance)
  - Early stopping (stops if val loss doesn't improve for `patience` epochs)
  - Checkpoint saving (saves best val-accuracy model to checkpoints/)
  - Epoch logging + training curve plot via matplotlib
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─── Training Utilities ───────────────────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility (PRD §7.3)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    return device


# ─── One Epoch Passes ─────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
) -> Tuple[float, float]:
    """
    Run one full pass over a DataLoader.

    Returns
    -------
    (avg_loss, accuracy)
    """
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels   = labels.to(device,   non_blocking=True)

            logits = model(features)                  # (B, 2)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


# ─── Main Training Function ───────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int       = 50,
    lr: float         = 3e-4,
    weight_decay: float = 1e-4,
    patience: int     = 10,
    checkpoint_name: str = "best_model.pt",
    seed: int         = 42,
) -> Dict[str, List[float]]:
    """
    Full training loop for binary speech sentiment classification.

    Parameters
    ----------
    model : nn.Module
        FCClassifier or CNNClassifier instance.
    train_loader, val_loader : DataLoader
        Training and validation DataLoaders.
    class_weights : torch.Tensor
        Tensor([w_negative, w_positive]) for weighted loss.
    epochs : int
        Maximum number of training epochs.
    lr : float
        Initial learning rate for AdamW.
    weight_decay : float
        L2 regularization coefficient.
    patience : int
        Early stopping patience (epochs without val improvement).
    checkpoint_name : str
        Filename to save best model checkpoint.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    set_seeds(seed)
    device = get_device()
    model  = model.to(device)

    # Weighted cross-entropy: penalizes misclassification of minority class more
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # AdamW — Adam with decoupled L2 weight decay (better than L2 in Adam)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # CosineAnnealingLR — smoothly decays LR from lr to 0 over `epochs` steps
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Training State ─────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc   = 0.0
    patience_count = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

    print(f"\n{'='*60}")
    print(f"  Training | Epochs: {epochs} | LR: {lr} | Patience: {patience}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Forward + backward pass on training set
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        # Evaluate on validation set (no grad)
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )

        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s"
        )

        # ── Checkpoint Best Model ──────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save({
                "epoch":      epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":    val_acc,
                "val_loss":   val_loss,
                "model_class": model.__class__.__name__,
            }, checkpoint_path)
            print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved.")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch} "
                      f"(patience={patience}). Best val_acc={best_val_acc:.4f}")
                break

    print(f"\n{'='*60}")
    print(f"  Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint saved to: {checkpoint_path}")
    print(f"{'='*60}\n")

    return history


# ─── Evaluation on Test Set ───────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Load the best checkpoint and evaluate on the test set.
    Prints accuracy, F1 score, and confusion matrix.

    Returns
    -------
    dict with keys: accuracy, f1_macro, f1_positive, f1_negative
    """
    device = get_device()

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Eval] Loaded checkpoint from epoch {ckpt['epoch']} "
              f"(val_acc={ckpt['val_acc']:.4f})")

    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0.0

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device, non_blocking=True)
            labels   = labels.to(device,   non_blocking=True)
            logits   = model(features)
            loss     = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy  = accuracy_score(all_labels, all_preds)
    f1_macro  = f1_score(all_labels, all_preds, average="macro")
    f1_pos    = f1_score(all_labels, all_preds, average="weighted", labels=[1])
    f1_neg    = f1_score(all_labels, all_preds, average="weighted", labels=[0])
    cm        = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  F1 Macro  : {f1_macro:.4f}")
    print(f"  F1 Positive: {f1_pos:.4f}")
    print(f"  F1 Negative: {f1_neg:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {cm}")
    print(f"{'='*50}\n")

    _plot_confusion_matrix(cm, ["Negative", "Positive"])

    return {
        "accuracy":     accuracy,
        "f1_macro":     f1_macro,
        "f1_positive":  f1_pos,
        "f1_negative":  f1_neg,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(history: Dict[str, List[float]], save_path: str = "training_curves.png") -> None:
    """
    Plot training + validation loss and accuracy curves, save to file.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SentiVoice — Training Curves", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#4A90E2", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#E24A4A", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch");  ax.set_ylabel("Loss")
    ax.set_title("Cross-Entropy Loss");  ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curve
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#4A90E2", linewidth=2)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Acc",   color="#E24A4A", linewidth=2, linestyle="--")
    ax.axhline(80, color="green", linewidth=1, linestyle=":", label="Target (80%)")
    ax.set_xlabel("Epoch");  ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy");  ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved to: {save_path}")


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                            save_path: str = "confusion_matrix.png") -> None:
    """Plot and save the confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved to: {save_path}")
