#!/usr/bin/env python3
"""
predict.py — SentiVoice Inference CLI
=======================================
Predict the sentiment of a speech audio file or live microphone input.

Usage:
  python predict.py --audio path/to/speech.wav
  python predict.py --audio path/to/speech.mp3 --model fc
  python predict.py --mic --seconds 5
  python predict.py --mic --seconds 3 --checkpoint checkpoints/my_model.pt
"""

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "SentiVoice — Predict sentiment (Positive/Negative) directly from speech audio.\n"
            "No speech-to-text. Uses MFCC features + PyTorch classifier."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input mode (mutually exclusive) ──────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio", "-a",
        type=str,
        metavar="FILE",
        help="Path to an audio file (.wav or .mp3) to predict.",
    )
    input_group.add_argument(
        "--mic", "-m",
        action="store_true",
        help="Record from microphone and predict in real time.",
    )

    # ── Model options ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "fc"],
        help="Model architecture to use. Default: cnn (recommended).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        metavar="PATH",
        help="Path to the trained model checkpoint (.pt). Default: checkpoints/best_model.pt",
    )

    # ── Microphone options ────────────────────────────────────────────────────
    parser.add_argument(
        "--seconds", "-s",
        type=float,
        default=5.0,
        help="Duration (seconds) to record from microphone. Default: 5.0",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full probability breakdown for both classes.",
    )

    return parser.parse_args()


def print_result(result: dict, verbose: bool = False) -> None:
    """Pretty-print the inference result."""
    sentiment  = result["sentiment"]
    confidence = result["confidence"] * 100
    label      = result["label"]

    # Sentiment icon
    icon = "😊" if label == 1 else "😠"

    print("\n" + "─" * 50)
    print(f"  {icon}  Sentiment  : {sentiment.upper()}")
    print(f"  📊  Confidence : {confidence:.1f}%")
    if verbose:
        print(f"\n  Class probabilities:")
        print(f"    Positive : {result['probs']['Positive'] * 100:.1f}%")
        print(f"    Negative : {result['probs']['Negative'] * 100:.1f}%")
    print("─" * 50 + "\n")


def main() -> None:
    args = parse_args()

    # ── Import here so --help works without heavy dependencies ────────────────
    try:
        from src.inference import SentiVoicePredictor
    except ImportError as e:
        print(f"[Error] Failed to import SentiVoice modules: {e}")
        print("Make sure you have run: pip install -r requirements.txt")
        sys.exit(1)

    # ── Load predictor ────────────────────────────────────────────────────────
    try:
        predictor = SentiVoicePredictor(
            checkpoint_path=args.checkpoint,
            model_type=args.model,
        )
    except FileNotFoundError as e:
        print(f"\n[Error] {e}\n")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if args.audio:
        print(f"\n[Predict] Analyzing: {args.audio}")
        try:
            result = predictor.predict_file(args.audio)
        except FileNotFoundError as e:
            print(f"[Error] {e}")
            sys.exit(1)
    else:
        print(f"\n[Predict] Microphone mode ({args.seconds}s recording)")
        result = predictor.predict_microphone(seconds=args.seconds)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[Predict] Inference completed in {elapsed_ms:.1f}ms")

    print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
