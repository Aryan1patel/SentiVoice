"""
src/dataset.py — RAVDESS Dataset Loader
========================================
Parses the RAVDESS filename convention to extract emotion labels,
maps them to binary sentiment (Positive / Negative), and provides
a PyTorch Dataset + DataLoader factory with stratified splits.

RAVDESS Filename Convention:
  Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
  e.g. 03-01-05-01-01-01-12.wav
       ^^ Modality 03 = audio-only speech

Emotion codes:
  01 - Neutral   → Positive
  02 - Calm      → Positive
  03 - Happy     → Positive
  04 - Sad       → Negative
  05 - Angry     → Negative
  06 - Fearful   → Negative
  07 - Disgusted → Negative
  08 - Surprised → Positive
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

from src.features import MFCCExtractor, get_feature_shape

# ─── Label Mapping ───────────────────────────────────────────────────────────
# RAVDESS emotion code → binary sentiment
# Positive = 1  (happy, calm, neutral, surprised)
# Negative = 0  (angry, fearful, sad, disgusted)
EMOTION_TO_SENTIMENT: Dict[int, int] = {
    1: 1,  # Neutral   → Positive
    2: 1,  # Calm      → Positive
    3: 1,  # Happy     → Positive
    4: 0,  # Sad       → Negative
    5: 0,  # Angry     → Negative
    6: 0,  # Fearful   → Negative
    7: 0,  # Disgusted → Negative
    8: 1,  # Surprised → Positive
}

EMOTION_NAMES: Dict[int, str] = {
    1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad",
    5: "Angry",   6: "Fearful", 7: "Disgusted", 8: "Surprised",
}

SENTIMENT_NAMES: Dict[int, str] = {0: "Negative", 1: "Positive"}

# ─── Filename Parser ──────────────────────────────────────────────────────────

def parse_ravdess_filename(filepath: str) -> Optional[Dict]:
    """
    Parse a RAVDESS audio filename and extract metadata.

    Parameters
    ----------
    filepath : str
        Full path to the .wav file.

    Returns
    -------
    dict with keys: path, modality, emotion_code, emotion_name, sentiment, actor
    None if the filename does not match the RAVDESS pattern.
    """
    basename = os.path.basename(filepath)
    match = re.match(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav", basename)
    if match is None:
        return None

    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor = (
        int(x) for x in match.groups()
    )

    # Only include audio-only speech files (modality 03)
    if modality != 3:
        return None

    if emotion_code not in EMOTION_TO_SENTIMENT:
        return None

    return {
        "path":         filepath,
        "modality":     modality,
        "emotion_code": emotion_code,
        "emotion_name": EMOTION_NAMES[emotion_code],
        "sentiment":    EMOTION_TO_SENTIMENT[emotion_code],
        "actor":        actor,
        "intensity":    intensity,
    }


def build_manifest(data_dir: str) -> pd.DataFrame:
    """
    Recursively scan data_dir for RAVDESS .wav files and build a manifest DataFrame.

    Parameters
    ----------
    data_dir : str
        Root directory containing RAVDESS audio files (can be nested in actor folders).

    Returns
    -------
    pd.DataFrame with columns: path, emotion_code, emotion_name, sentiment, actor, intensity
    """
    records = []
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            fpath  = os.path.join(root, fname)
            record = parse_ravdess_filename(fpath)
            if record is not None:
                records.append(record)

    if not records:
        raise FileNotFoundError(
            f"No RAVDESS .wav files found in '{data_dir}'.\n"
            "Please ensure the dataset is downloaded and extracted."
        )

    df = pd.DataFrame(records)
    print(f"[Dataset] Found {len(df)} clips | "
          f"Positive: {(df.sentiment == 1).sum()} | "
          f"Negative: {(df.sentiment == 0).sum()}")
    return df


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class RAVDESSDataset(Dataset):
    """
    PyTorch Dataset for RAVDESS binary sentiment classification.

    Each item is a tuple (features, label) where:
      - features: float32 tensor of shape (3*N_MFCC, MAX_LEN_FRAMES)
      - label:    long tensor — 0 (Negative) or 1 (Positive)

    Parameters
    ----------
    manifest : pd.DataFrame
        Rows from build_manifest() corresponding to this split.
    augment : bool
        Whether to apply data augmentation during feature extraction.
    cache : bool
        If True, precompute and cache all features in memory.
        Speeds up training but requires ~(n_samples * 3*40*200*4) bytes RAM.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        augment: bool = False,
        cache: bool = True,
    ):
        self.manifest  = manifest.reset_index(drop=True)
        self.extractor = MFCCExtractor(augment=augment)
        self.cache     = cache
        self._cache: Dict[int, torch.Tensor] = {}

        if cache and not augment:
            # Pre-extract all features into RAM
            print(f"[Dataset] Pre-caching {len(self.manifest)} features...")
            for idx in tqdm(range(len(self.manifest)), ncols=80):
                self._cache[idx] = self._extract(idx)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx in self._cache:
            features = self._cache[idx]
        else:
            features = self._extract(idx)

        label = torch.tensor(self.manifest.iloc[idx]["sentiment"], dtype=torch.long)
        return features, label

    def _extract(self, idx: int) -> torch.Tensor:
        row   = self.manifest.iloc[idx]
        feats = self.extractor.extract(row["path"])          # (3*N_MFCC, MAX_LEN)
        return torch.tensor(feats, dtype=torch.float32)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for CrossEntropy loss.
        Helps handle class imbalance (PRD §11).

        Returns tensor([w_negative, w_positive]).
        """
        counts = self.manifest["sentiment"].value_counts().sort_index()
        total  = len(self.manifest)
        weights = total / (len(counts) * counts.values)
        return torch.tensor(weights, dtype=torch.float32)


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,          # 0 = main process only; avoids macOS spawn issues
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    augment_train: bool = True,
    cache: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build stratified train/val/test DataLoaders from a RAVDESS data directory.

    Parameters
    ----------
    data_dir : str
        Path to folder containing RAVDESS .wav files.
    batch_size : int
        Samples per gradient step.
    num_workers : int
        Parallel worker processes for data loading.
    val_size, test_size : float
        Fractions of the dataset for validation and test splits.
    random_state : int
        Fixed seed for reproducible splits (PRD §7.3).
    augment_train : bool
        Apply augmentation to training samples only.
    cache : bool
        Cache features in memory (faster training, more RAM).

    Returns
    -------
    (train_loader, val_loader, test_loader, class_weights)
    """
    # Set seeds for reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    manifest = build_manifest(data_dir)

    # Stratified split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        manifest,
        test_size=val_size + test_size,
        stratify=manifest["sentiment"],
        random_state=random_state,
    )
    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        stratify=temp_df["sentiment"],
        random_state=random_state,
    )

    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = RAVDESSDataset(train_df, augment=augment_train, cache=cache and not augment_train)
    val_ds   = RAVDESSDataset(val_df,   augment=False,          cache=cache)
    test_ds  = RAVDESSDataset(test_df,  augment=False,          cache=cache)

    class_weights = train_ds.get_class_weights()

    # pin_memory speeds up host→GPU transfers but is only supported on CUDA.
    # On MPS (Apple Silicon) and CPU it triggers warnings and has no benefit.
    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, class_weights
