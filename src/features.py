"""
src/features.py — MFCC Feature Extraction Module
=================================================
Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio files.
MFCCs are a compact representation of the spectral envelope of speech:
  1. Frame the signal into overlapping windows
  2. Apply the Discrete Fourier Transform (DFT) to each frame
  3. Map frequencies onto the Mel scale (mimics human hearing, log-spaced)
  4. Apply log compression to the Mel filter-bank energies
  5. Apply Discrete Cosine Transform (DCT) to decorrelate features → MFCCs

We also compute delta (velocity) and delta-delta (acceleration) features,
giving the model temporal dynamics of the speech signal.

Total feature vector: 40 MFCCs + 40 deltas + 40 delta-deltas = 120 features per frame.
"""

import numpy as np
import librosa
import librosa.effects
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

# ─── Constants ───────────────────────────────────────────────────────────────
SAMPLE_RATE     = 22_050   # Hz — default librosa sample rate
N_MFCC          = 40       # Number of MFCC coefficients (PRD spec)
HOP_LENGTH      = 512      # Samples between frames (~23ms at 22050 Hz)
N_FFT           = 2048     # FFT window size (~93ms at 22050 Hz)
PRE_EMPHASIS    = 0.97     # Pre-emphasis filter coefficient
MAX_LEN_FRAMES  = 200      # Pad/truncate all clips to this many frames
                           # (200 × 512 / 22050 ≈ 4.65 seconds — covers most RAVDESS clips)

# ─── Augmentation Parameters ─────────────────────────────────────────────────
AUG_TIME_STRETCH_RATES = [0.9, 1.1]  # ±10% speed change
AUG_PITCH_SHIFT_STEPS  = [-2, 2]     # ±2 semitones
AUG_NOISE_SNR_DB       = 20          # Signal-to-noise ratio for Gaussian noise


class MFCCExtractor:
    """
    Extracts fixed-length MFCC feature matrices from audio files.

    Parameters
    ----------
    n_mfcc : int
        Number of MFCC coefficients to extract per frame.
    max_len : int
        Number of time frames to pad/truncate each clip to.
        Ensures all samples have identical shape for batching.
    sr : int
        Target sample rate. Audio is resampled to this if needed.
    augment : bool
        If True, randomly applies time-stretch, pitch-shift, or noise.
    """

    def __init__(
        self,
        n_mfcc: int = N_MFCC,
        max_len: int = MAX_LEN_FRAMES,
        sr: int = SAMPLE_RATE,
        augment: bool = False,
    ):
        self.n_mfcc  = n_mfcc
        self.max_len = max_len
        self.sr      = sr
        self.augment = augment

    # ── Public API ─────────────────────────────────────────────────────────

    def extract(self, audio_path: str) -> np.ndarray:
        """
        Load an audio file and return its normalized MFCC feature matrix.

        Returns
        -------
        np.ndarray of shape (3 * n_mfcc, max_len)
            Rows: [mfcc | delta | delta-delta], Cols: time frames.
            Values are zero-mean, unit-variance normalized.
        """
        # Step 1: Load raw waveform (float32, mono, resampled to target SR)
        waveform, _ = librosa.load(audio_path, sr=self.sr, mono=True)

        # Step 2: Optional data augmentation (training only)
        if self.augment:
            waveform = self._augment(waveform)

        # Step 3: Pre-emphasis filter — boosts high frequencies before feature extraction
        # y[t] = x[t] - α * x[t-1]  (α = PRE_EMPHASIS ≈ 0.97)
        waveform = np.append(waveform[0], waveform[1:] - PRE_EMPHASIS * waveform[:-1])

        # Step 4: Extract raw MFCCs
        # librosa applies Mel filterbank + log + DCT internally
        mfccs = librosa.feature.mfcc(
            y=waveform,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )  # shape: (n_mfcc, T)

        # Step 5: Compute delta features (first derivative over time)
        # Δ[t] ≈ (mfcc[t+1] - mfcc[t-1]) / 2  (captures velocity of features)
        delta  = librosa.feature.delta(mfccs)           # shape: (n_mfcc, T)

        # Step 6: Compute delta-delta features (second derivative over time)
        # ΔΔ captures acceleration — useful for modeling speech dynamics
        delta2 = librosa.feature.delta(mfccs, order=2)  # shape: (n_mfcc, T)

        # Step 7: Stack all features → (3*n_mfcc, T)
        features = np.vstack([mfccs, delta, delta2])

        # Step 8: Pad or truncate to fixed length
        features = self._fix_length(features)  # shape: (3*n_mfcc, max_len)

        # Step 9: Normalize — zero-mean, unit-variance across the feature axis
        features = self._normalize(features)

        return features.astype(np.float32)

    def extract_from_array(self, waveform: np.ndarray) -> np.ndarray:
        """Same as extract() but takes a pre-loaded waveform array."""
        waveform = np.append(waveform[0], waveform[1:] - PRE_EMPHASIS * waveform[:-1])
        mfccs   = librosa.feature.mfcc(y=waveform, sr=self.sr, n_mfcc=self.n_mfcc,
                                        n_fft=N_FFT, hop_length=HOP_LENGTH)
        delta   = librosa.feature.delta(mfccs)
        delta2  = librosa.feature.delta(mfccs, order=2)
        features = np.vstack([mfccs, delta, delta2])
        features = self._fix_length(features)
        features = self._normalize(features)
        return features.astype(np.float32)

    # ── Private Helpers ────────────────────────────────────────────────────

    def _fix_length(self, features: np.ndarray) -> np.ndarray:
        """Pad with zeros or truncate along the time axis to self.max_len."""
        n_frames = features.shape[1]
        if n_frames < self.max_len:
            # Zero-pad on the right
            pad_width = self.max_len - n_frames
            features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
        else:
            # Truncate to max_len frames
            features = features[:, : self.max_len]
        return features

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Zero-mean, unit-variance normalization across the time axis per feature row.
        Prevents dominant features from drowning out weaker ones.
        """
        mean = features.mean(axis=1, keepdims=True)
        std  = features.std(axis=1, keepdims=True) + 1e-8  # add eps to avoid /0
        return (features - mean) / std

    def _augment(self, waveform: np.ndarray) -> np.ndarray:
        """
        Randomly apply one of three augmentations:
          - Time stretching (±10% speed)
          - Pitch shifting (±2 semitones)
          - Adding Gaussian noise at SNR=20dB

        Augmentation is applied with p=0.5 per type.
        """
        rng = np.random.default_rng()

        # Time stretch — changes duration without affecting pitch
        if rng.random() < 0.5:
            rate = rng.choice(AUG_TIME_STRETCH_RATES)
            waveform = librosa.effects.time_stretch(waveform, rate=rate)

        # Pitch shift — changes pitch without affecting duration
        if rng.random() < 0.5:
            steps = rng.choice(AUG_PITCH_SHIFT_STEPS)
            waveform = librosa.effects.pitch_shift(waveform, sr=self.sr, n_steps=int(steps))

        # Gaussian noise at 20dB SNR
        if rng.random() < 0.5:
            signal_power = np.mean(waveform ** 2)
            noise_power  = signal_power / (10 ** (AUG_NOISE_SNR_DB / 10))
            noise        = rng.normal(0, np.sqrt(noise_power), len(waveform))
            waveform     = waveform + noise.astype(np.float32)

        return waveform


def get_feature_shape() -> Tuple[int, int]:
    """Returns the feature matrix shape: (3 * N_MFCC, MAX_LEN_FRAMES)."""
    return (3 * N_MFCC, MAX_LEN_FRAMES)
