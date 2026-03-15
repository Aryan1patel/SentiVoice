#!/usr/bin/env python3
"""
train_model.py — SentiVoice Training Entry Point
===================================================
Full training pipeline: data loading → model creation → training → evaluation.

Usage:
  python train_model.py --data-dir data/
  python train_model.py --data-dir data/ --model cnn --epochs 50 --lr 3e-4
  python train_model.py --data-dir data/ --model fc  --epochs 30 --batch-size 64
  python train_model.py --data-dir data/ --epochs 50 --no-augment  # skip augmentation
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_model.py",
        description=(
            "SentiVoice Training Pipeline\n"
            "Trains a binary speech sentiment classifier on RAVDESS audio data.\n"
            "\nPipeline: MFCC extraction → DataLoader → PyTorch training loop → evaluation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        required=False,
        default="dataset/archive-10",
        metavar="DIR",
        help="Path to directory containing RAVDESS .wav files. Default: dataset/archive-10",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "fc"],
        help="Model architecture. 'cnn' = 1D CNN (recommended), 'fc' = fully connected. Default: cnn",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate (0.0–1.0). Default: 0.5",
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs. Default: 50",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate for AdamW. Default: 3e-4",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decay coefficient. Default: 1e-4",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training. Default: 32",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement). Default: 10",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )

    # ── Data options ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of data for validation set. Default: 0.15",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of data for test set. Default: 0.15",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes. Default: 4",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training data augmentation.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable in-memory feature caching (saves RAM, slower training).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Filename for the saved checkpoint. Default: best_model.pt",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate data dir ─────────────────────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        print(f"[Error] Data directory not found: '{args.data_dir}'")
        print("Please ensure the dataset is placed at the specified path.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  SentiVoice — Speech Sentiment Classifier Training")
    print("=" * 60)
    print(f"  Data   : {args.data_dir}")
    print(f"  Model  : {args.model.upper()} | Dropout: {args.dropout}")
    print(f"  Epochs : {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"  Seed   : {args.seed} | Augment: {not args.no_augment}")
    print("=" * 60 + "\n")

    # ── Imports (after prints so help is fast) ────────────────────────────────
    try:
        from src.dataset import get_dataloaders
        from src.model   import build_model
        from src.train   import train, evaluate, plot_training_curves
    except ImportError as e:
        print(f"[Error] Import failed: {e}\nRun: pip install -r requirements.txt")
        sys.exit(1)

    # ── Step 1: Load Data ─────────────────────────────────────────────────────
    print("[Step 1/4] Building DataLoaders...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        augment_train=not args.no_augment,
        cache=not args.no_cache,
    )
    print(f"  Class weights: Negative={class_weights[0]:.3f}, Positive={class_weights[1]:.3f}\n")

    # ── Step 2: Build Model ───────────────────────────────────────────────────
    print("[Step 2/4] Building model...")
    model = build_model(model_type=args.model, dropout=args.dropout)

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\n[Step 3/4] Training...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_name=args.checkpoint_name,
        seed=args.seed,
    )

    # Plot training curves
    plot_training_curves(history, save_path="training_curves.png")

    # ── Step 4: Evaluate on Test Set ──────────────────────────────────────────
    print("[Step 4/4] Evaluating on test set...")
    checkpoint_path = os.path.join("checkpoints", args.checkpoint_name)
    metrics = evaluate(
        model=model,
        test_loader=test_loader,
        class_weights=class_weights,
        checkpoint_path=checkpoint_path,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    status_acc = "✓ PASS" if metrics["accuracy"] >= 0.80 else "✗ Below target"
    status_f1  = "✓ PASS" if metrics["f1_macro"]  >= 0.78 else "✗ Below target"
    print(f"  Accuracy   : {metrics['accuracy']*100:.1f}%   (target ≥ 80%) {status_acc}")
    print(f"  F1 Macro   : {metrics['f1_macro']:.4f}  (target ≥ 0.78) {status_f1}")
    print(f"  Checkpoint : checkpoints/{args.checkpoint_name}")
    print(f"  Curves     : training_curves.png")
    print(f"  Confusion  : confusion_matrix.png")
    print("\n  Run inference:")
    print(f"    python predict.py --audio your_speech.wav")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
