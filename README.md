# 🎙️ SentiVoice — Speech Sentiment Analysis

> **"A deep learning system that predicts emotional sentiment directly from raw speech, bypassing speech-to-text entirely."**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

SentiVoice predicts binary emotional sentiment (**Positive** vs **Negative**) directly from recorded voice by analyzing pitch, tone, and rhythm. 

By extracting Mel-Frequency Cepstral Coefficients (**MFCCs**) and analyzing them with a custom **PyTorch 1D CNN**, it achieves over **92% validation accuracy** without transcribing a single word.

---

<img width="1709" height="1087" alt="image" src="https://github.com/user-attachments/assets/dade6a69-9943-4421-9342-c9d2a9de5d65" />


## ⚡ Quick Start

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the Dataset
1. Download the [RAVDESS Audio Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).
2. Extract the `Audio_Speech_Actors_01-24` folders into `dataset/archive-10/`.

### 3. Train the Model
```bash
python train_model.py
```
*(This automatically builds the CNN model, trains it using early stopping, and saves the best model to `checkpoints/best_model.pt`.)*

### 4. Run Predictions
You can test the trained model on an audio file or directly using your microphone!

```bash
# Test an audio file
python predict.py --audio path/to/speech.wav --verbose

# Test your own voice (records for 5 seconds)
python predict.py --mic --seconds 5
```

**Example Output:**
```
──────────────────────────────────────────────────
  😊  Sentiment  : POSITIVE
  📊  Confidence : 87.4%
──────────────────────────────────────────────────
```

### 5. Start the REST API
Want to connect a frontend? Start the local Flask server:
```bash
python app.py
```
Send a POST request to `http://localhost:5000/predict` with an audio file to get instant JSON sentiment predictions.

---

## 🧠 How it Works

Traditional NLP sentiment pipelines require a Speech-to-Text model followed by a Text Sentiment model. This is slow and loses critical emotional cues (like sarcasm, yelling, or crying).

Instead, SentiVoice uses **Digital Signal Processing**:
1. Feeds the raw `.wav` file through a **Pre-Emphasis filter** and extracts 120 features per frame (MFCCs + Deltas + Delta-Deltas) using `librosa`.
2. Normalizes the shape and handles variable-length audio via padding/truncation.
3. Passes the spectral fingerprint into a highly efficient **1D Convolutional Neural Network (CNN)**.

---

## 📊 Dataset (RAVDESS)

The dataset consists of 24 professional actors demonstrating 8 distinct emotions. For this binary classifier, we grouped them into:

- **Positive (1):** Happy, Calm, Neutral, Surprised
- **Negative (0):** Angry, Fearful, Sad, Disgusted

During training, we use **Data Augmentation** (pitch shifting, time stretching, and background noise addition) to prevent overfitting and make the model robust to real-world environments.

---

## 📦 Tech Stack

- **Deep Learning:** PyTorch, TorchAudio
- **Audio Processing:** Librosa, SoundDevice
- **Data & Setup:** Scikit-Learn, Pandas, NumPy
- **API Server:** Flask
