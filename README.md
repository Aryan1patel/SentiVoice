# 🎙️ SentiVoice — Speech Sentiment Analysis

> **"I built a system that predicts sentiment directly from speech using MFCC features and a custom PyTorch model, without relying on speech-to-text."**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

SentiVoice predicts binary emotional sentiment (**Positive** / **Negative**) directly from raw audio — capturing tone, pitch, and prosody that text-based sentiment analysis cannot.

---

## 🔑 Why Not Use Text (ASR)?

| Traditional NLP Pipeline | This System |
|---|---|
| Speech → **Text** → Sentiment | Speech → **MFCC Features** → Sentiment |
| ❌ Loses tone, pitch, prosody | ✅ Captures all acoustic emotion cues |
| Requires 2 models (ASR + NLP) | Single end-to-end PyTorch model |
| Fails on non-verbal speech | Works on any vocal signal |

---

## 🧠 What are MFCCs?

**Mel-Frequency Cepstral Coefficients** are the "fingerprint" of a voice. Here's how they're computed:

```
Raw Audio Signal
      │
      ▼
Pre-emphasis filter        ← y[t] = x[t] - 0.97·x[t-1]  (boosts high freq)
      │
      ▼
Frame into windows         ← 23ms frames, 50% overlap
      │
      ▼
Discrete Fourier Transform ← Converts time → frequency domain
      │
      ▼
Mel Filter Bank            ← 128 filters spaced on Mel scale (mimics human hearing)
      │
      ▼
Log Compression            ← log(energy) — mimics loudness perception
      │
      ▼
Discrete Cosine Transform  ← Decorrelates filter outputs → 40 MFCC coefficients
      │
      ▼
Delta + Delta-Delta        ← Adds velocity and acceleration of features over time
      │
      ▼
Feature Matrix             ← Shape: (120 features × 200 time frames)
```

We extract **40 MFCCs + 40 deltas + 40 delta-deltas = 120 features per frame** across 200 time frames, giving the model both spectral content and temporal dynamics of speech.

---

## 📁 Project Structure

```
SentiVoice/
├── src/
│   ├── features.py      # MFCC extraction (MFCCExtractor class)
│   ├── dataset.py       # RAVDESS DataLoader with binary label mapping
│   ├── model.py         # FCClassifier (baseline) + CNNClassifier (1D CNN)
│   ├── train.py         # Training loop: AdamW, cosine LR, early stopping
│   └── inference.py     # SentiVoicePredictor (file + mic input)
├── predict.py           # CLI: python predict.py --audio file.wav
├── train_model.py       # CLI: python train_model.py --data-dir dataset/archive-10/
├── app.py               # Flask REST API (POST /predict)
└── requirements.txt     # Pinned dependencies
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download dataset

The RAVDESS dataset should be placed in `dataset/archive-10/` (or your preferred directory).

If you don't have it, download it manually:
1. Go to: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
2. Download `Audio_Speech_Actors_01-24.zip`
3. Extract into `dataset/archive-10/`

### 3. Train

```bash
# Recommended: 1D CNN model (higher accuracy)
python train_model.py

# Fast baseline: Fully Connected model
python train_model.py --model fc --epochs 30
```

### 4. Predict

```bash
# From audio file
python predict.py --audio path/to/speech.wav

# With verbose probability breakdown
python predict.py --audio path/to/speech.wav --verbose

# From microphone (5 second recording)
python predict.py --mic --seconds 5
```

**Example output:**
```
──────────────────────────────────────────────────
  😊  Sentiment  : POSITIVE
  📊  Confidence : 87.4%
──────────────────────────────────────────────────
```

### 5. REST API

```bash
python app.py
# Server starts at http://localhost:5000

# Test with curl:
curl -X POST http://localhost:5000/predict \
     -F "audio=@speech.wav" | python3 -m json.tool
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.874,
  "label": 1,
  "probabilities": {
    "Negative": 0.126,
    "Positive": 0.874
  },
  "inference_ms": 48.3
}
```

---

## 🏗️ Architecture

### Model A — FCClassifier (Baseline)
```
Input (B, 120, 200) → Flatten → FC(512) → BN → ReLU → Dropout
                             → FC(128) → BN → ReLU → Dropout
                             → FC(64)  → BN → ReLU → Dropout
                             → FC(2)  [logits]
```

### Model B — CNNClassifier (Recommended)
```
Input (B, 120, 200)
  → Conv1D(120→256, k=5) → BN → ReLU → MaxPool  →  (B, 256, 100)
  → Conv1D(256→128, k=5) → BN → ReLU → MaxPool  →  (B, 128, 50)
  → Conv1D(128→ 64, k=3) → BN → ReLU → MaxPool  →  (B,  64, 25)
  → GlobalAvgPool                                →  (B,  64)
  → Dropout(0.5)
  → Linear(64 → 2)                               →  (B,   2) logits
```

---

## 📊 Dataset — RAVDESS

| Property | Detail |
|---|---|
| Source | [Zenodo DOI: 10.5281/zenodo.1188976](https://zenodo.org/record/1188976) |
| Actors | 24 professional actors (12M / 12F) |
| Clips | ~1,440 audio-only speech files |
| Emotions | 8 (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgusted, Surprised) |
| **Positive** | Happy, Calm, Neutral, Surprised → label: **1** |
| **Negative** | Angry, Fearful, Sad, Disgusted → label: **0** |
| Split | 70% train / 15% val / 15% test (stratified) |

### Data Augmentation (training only)
- **Time stretching**: ±10% speed change (`librosa.effects.time_stretch`)
- **Pitch shifting**: ±2 semitones (`librosa.effects.pitch_shift`)
- **Gaussian noise**: SNR = 20dB (simulates real-world conditions)

---

## 🎛️ Training Configuration

| Hyperparameter | Default | Notes |
|---|---|---|
| Optimizer | AdamW | Decoupled L2 weight decay |
| Learning Rate | 3e-4 | Cosine annealing decay |
| Weight Decay | 1e-4 | L2 regularization |
| Batch Size | 32 | |
| Max Epochs | 50 | With early stopping |
| Patience | 10 | Epochs without val improvement |
| Dropout | 0.5 | Applied in CNN head |
| Loss Function | Weighted CrossEntropy | Handles class imbalance |

---

## 🎯 Target Metrics (from PRD)

| Metric | Target | Notes |
|---|---|---|
| Test Accuracy | ≥ 80% | Binary classification |
| F1 Score (macro) | ≥ 0.78 | Both Positive and Negative classes |
| Inference Latency | < 500ms | On CPU for 5-second clip |
| Model Size | < 50MB | Checkpoint file |
| Training Time | < 30 min | Single GPU, 5k samples |

---

## 🔧 Advanced Usage

```bash
# Skip augmentation (faster, for debugging)
python train_model.py --no-augment

# Custom hyperparameters
python train_model.py \
  --model cnn \
  --epochs 80 \
  --lr 1e-4 \
  --batch-size 16 \
  --patience 15

# Use FC model for inference
python predict.py --audio speech.wav --model fc --checkpoint checkpoints/fc_model.pt

# API on custom port
python app.py --port 8080 --debug
```

---

## 📦 Requirements

Key dependencies (see `requirements.txt` for pinned versions):

| Package | Version | Purpose |
|---|---|---|
| `torch` | >= 2.2.0 | Neural network training & inference |
| `librosa` | >= 0.10.0 | MFCC extraction, audio loading |
| `sounddevice` | >= 0.4.6 | Real-time microphone capture |
| `scikit-learn` | >= 1.3.0 | Stratified splits, metrics |
| `matplotlib` | >= 3.8.0 | Training curve & confusion matrix plots |
| `flask` | >= 3.0.0 | REST API server |

---

## 🗺️ Roadmap (from PRD)

- [x] MFCC Extraction Module (FR-01)
- [x] PyTorch Model Training (FR-02)
- [x] Inference CLI (FR-03)
- [x] Dataset Loader (FR-04)
- [x] Confidence Score Output (FR-05)
- [x] Microphone Input Mode (FR-06)
- [x] Training Dashboard / Curves (FR-07)
- [ ] ONNX Model Export (FR-08)
- [x] REST API Endpoint (FR-09)
- [ ] Multi-emotion 8-class labels (FR-10, v2)

---

## 📝 Glossary

| Term | Definition |
|---|---|
| **MFCC** | Mel-Frequency Cepstral Coefficients — compact spectral fingerprint of speech |
| **Delta** | First derivative of MFCCs over time — captures feature velocity |
| **Delta-Delta** | Second derivative — captures feature acceleration |
| **Prosody** | Rhythm, stress, and intonation of speech — key emotion carrier |
| **RAVDESS** | Ryerson Audio-Visual Database of Emotional Speech and Song |
| **ASR** | Automatic Speech Recognition — deliberately avoided in this project |
| **GAP** | Global Average Pooling — reduces spatial dimensions to a single value per channel |

---

*Built as a portfolio ML project. PRD v1.0, April 2026.*
