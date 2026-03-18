"""
src/inference.py — Inference Engine
=====================================
Provides prediction functions for both file-based and real-time microphone input.

Usage:
    from src.inference import SentiVoicePredictor
    predictor = SentiVoicePredictor("checkpoints/best_model.pt", model_type="cnn")
    result = predictor.predict_file("audio.wav")
    # → {"sentiment": "Positive", "confidence": 0.874, "label": 1}
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional

from src.features import MFCCExtractor, SAMPLE_RATE
from src.model import build_model
from src.dataset import SENTIMENT_NAMES


class SentiVoicePredictor:
    """
    Unified inference engine for SentiVoice speech sentiment prediction.

    Loads a trained PyTorch checkpoint and exposes methods to predict
    sentiment from audio files or live microphone recordings.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt checkpoint saved during training.
    model_type : str
        'fc' or 'cnn' — must match the architecture used during training.
    device : str or None
        Force a specific device ('cpu', 'cuda', 'mps'). Auto-detects if None.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "cnn",
        device: Optional[str] = None,
    ):
        self.extractor = MFCCExtractor(augment=False)

        # ── Device selection ────────────────────────────────────────────
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ── Load model ──────────────────────────────────────────────────
        self.model = build_model(model_type)
        self._load_checkpoint(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[Inference] Model: {self.model.__class__.__name__} | "
              f"Device: {self.device} | Checkpoint: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load weights from a training checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: '{checkpoint_path}'.\n"
                "Train the model first: python train_model.py --data-dir data/"
            )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        epoch    = ckpt.get("epoch", "?")
        val_acc  = ckpt.get("val_acc", 0.0)
        print(f"[Inference] Loaded checkpoint (epoch={epoch}, val_acc={val_acc:.4f})")

    # ── Public Prediction Methods ────────────────────────────────────────────

    def predict_file(self, audio_path: str) -> Dict:
        """
        Predict sentiment from an audio file (.wav or .mp3).

        Parameters
        ----------
        audio_path : str
            Path to the audio file.

        Returns
        -------
        dict:
            sentiment  : str  — 'Positive' or 'Negative'
            confidence : float — softmax probability of the predicted class (0–1)
            label      : int  — 0 (Negative) or 1 (Positive)
            probs      : dict — {'Negative': float, 'Positive': float}
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

        features = self.extractor.extract(audio_path)
        return self._run_inference(features)

    def predict_microphone(self, seconds: float = 5.0) -> Dict:
        """
        Record `seconds` of audio from the default microphone and predict sentiment.

        Requires the `sounddevice` package.

        Parameters
        ----------
        seconds : float
            Duration to record in seconds.

        Returns
        -------
        Same dict format as predict_file().
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("Install sounddevice: pip install sounddevice")

        print(f"[Mic] Recording for {seconds:.1f} second(s)... (speak now)")
        waveform = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("[Mic] Recording complete. Processing...")

        waveform = waveform.flatten()  # (N,)
        features = self.extractor.extract_from_array(waveform)
        return self._run_inference(features)

    # ── Internal Inference ───────────────────────────────────────────────────

    def _run_inference(self, features: np.ndarray) -> Dict:
        """
        Run the model on a pre-extracted feature array.

        Parameters
        ----------
        features : np.ndarray of shape (3*N_MFCC, MAX_LEN_FRAMES)

        Returns
        -------
        dict with 'sentiment', 'confidence', 'label', 'probs'
        """
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 120, 200)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)                   # (1, 2)
            probs  = F.softmax(logits, dim=1).squeeze()   # (2,)

        label      = probs.argmax().item()
        confidence = probs[label].item()
        sentiment  = SENTIMENT_NAMES[label]

        return {
            "sentiment":  sentiment,
            "confidence": round(confidence, 4),
            "label":      label,
            "probs": {
                "Negative": round(probs[0].item(), 4),
                "Positive": round(probs[1].item(), 4),
            },
        }

    def __repr__(self) -> str:
        return (f"SentiVoicePredictor("
                f"model={self.model.__class__.__name__}, "
                f"device={self.device})")
