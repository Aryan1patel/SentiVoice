"""
app.py — SentiVoice Flask REST API
====================================
Exposes a REST API endpoint for speech sentiment prediction.
Upload an audio file and receive JSON with the sentiment + confidence score.

Endpoints:
  GET  /             — Health check + API info
  POST /predict      — Predict sentiment from uploaded audio file
  GET  /model-info   — Return loaded model metadata

Usage:
  python app.py
  python app.py --port 8080 --model fc

Example cURL:
  curl -X POST http://localhost:5000/predict \\
       -F "audio=@path/to/speech.wav" | python3 -m json.tool
"""

import os
import sys
import time
import argparse
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ── Allowed upload extensions ─────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Global predictor (loaded at startup)
predictor = None
model_info = {}


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    """Health check endpoint — returns API status and usage info."""
    return jsonify({
        "status":  "ok",
        "service": "SentiVoice Speech Sentiment API",
        "version": "1.0.0",
        "model":   model_info.get("model_class", "unknown"),
        "endpoints": {
            "POST /predict":    "Upload audio file → get sentiment + confidence",
            "GET  /model-info": "Return model metadata",
        },
        "usage": {
            "method":  "POST",
            "url":     "/predict",
            "body":    "multipart/form-data",
            "field":   "audio",
            "accepts": list(ALLOWED_EXTENSIONS),
        },
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict sentiment from an uploaded audio file.

    Request: multipart/form-data with field 'audio' containing an audio file.
    Response: JSON with sentiment, confidence, and probabilities.
    """
    global predictor

    if predictor is None:
        return jsonify({"error": "Model not loaded. Check server startup logs."}), 503

    # ── Validate request ──────────────────────────────────────────────────────
    if "audio" not in request.files:
        return jsonify({"error": "No audio file found. Send as form-data field 'audio'."}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "Empty filename. Please upload a valid audio file."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
        }), 415

    # ── Save to temp file and run inference ───────────────────────────────────
    suffix = "." + secure_filename(file.filename).rsplit(".", 1)[-1]
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_file = tmp.name

        t0 = time.perf_counter()
        result = predictor.predict_file(tmp_file)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return jsonify({
            "sentiment":    result["sentiment"],
            "confidence":   result["confidence"],
            "label":        result["label"],
            "probabilities": result["probs"],
            "inference_ms": round(elapsed_ms, 2),
            "filename":     secure_filename(file.filename),
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Always clean up the temp file
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)


@app.route("/model-info", methods=["GET"])
def get_model_info():
    """Return metadata about the loaded model."""
    return jsonify(model_info), 200


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": f"File too large. Maximum size: {MAX_CONTENT_LENGTH // (1024*1024)}MB"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found. See GET / for available endpoints."}), 404


# ── Startup ───────────────────────────────────────────────────────────────────

def load_predictor(checkpoint_path: str, model_type: str) -> None:
    """Load the SentiVoice predictor at server startup."""
    global predictor, model_info
    try:
        from src.inference import SentiVoicePredictor
        import torch

        predictor  = SentiVoicePredictor(checkpoint_path=checkpoint_path, model_type=model_type)
        ckpt       = torch.load(checkpoint_path, map_location="cpu")
        model_info = {
            "model_class":     predictor.model.__class__.__name__,
            "checkpoint_path": checkpoint_path,
            "checkpoint_epoch": ckpt.get("epoch", "?"),
            "val_accuracy":    ckpt.get("val_acc", None),
            "device":          str(predictor.device),
        }
        print(f"[API] Predictor loaded: {model_info}")
    except FileNotFoundError:
        print(f"[Warning] Checkpoint not found: '{checkpoint_path}'")
        print("[Warning] /predict endpoint will return 503 until a model is trained.")
        print("[Warning] Run: python train_model.py --data-dir data/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentiVoice Flask REST API")
    parser.add_argument("--port",       type=int,   default=5000,       help="Server port. Default: 5000")
    parser.add_argument("--host",       type=str,   default="0.0.0.0",  help="Bind host. Default: 0.0.0.0")
    parser.add_argument("--model",      type=str,   default="cnn",      choices=["cnn", "fc"])
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/best_model.pt")
    parser.add_argument("--debug",      action="store_true",            help="Enable Flask debug mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 55)
    print("  SentiVoice REST API")
    print("=" * 55)

    load_predictor(checkpoint_path=args.checkpoint, model_type=args.model)

    print(f"  Starting server on http://{args.host}:{args.port}")
    print(f"  POST /predict  — Upload audio file for sentiment prediction")
    print("=" * 55 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
