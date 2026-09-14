"""Flask application entry point for the NMT inference API.

Endpoints
---------
GET  /health              — liveness check
POST /translate           — translate a single text string
POST /translate/attention — translate and return attention weights
"""

import json
import logging
import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

# Allow importing src/nmt from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import inference_service
from nmt.utils.logging import setup_logging

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Startup: initialize the inference service once when the app launches
# ---------------------------------------------------------------------------


def _initialize_service() -> None:
    """Attempt to load the model at startup. Fails gracefully if paths are not set."""
    try:
        inference_service.initialize()
        logger.info("Inference service initialized successfully.")
    except EnvironmentError as e:
        logger.warning(
            "Inference service not initialized (model paths not configured): %s", e
        )
    except Exception as e:
        logger.error("Failed to initialize inference service: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness and readiness check.

    Returns:
        JSON with status, model_ready flag, and version.
    """
    return jsonify(
        {
            "status": "ok",
            "model_ready": inference_service.is_ready(),
            "version": "0.1.0",
        }
    )


@app.post("/translate")
def translate():
    """Translate a source text string.

    Request body (JSON):
        {
            "text": "<source text>"
        }

    Response (JSON):
        {
            "translation": "<translated text>"
        }

    Error responses:
        400 — missing or empty 'text' field
        503 — model not initialized
        500 — unexpected server error
    """
    if not inference_service.is_ready():
        return jsonify({"error": "Model not ready. Check server logs."}), 503

    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "'text' field is required and must be non-empty."}), 400

    text = data["text"].strip()
    logger.debug("Translating: %r", text[:100])

    try:
        translation = inference_service.translate(text)
        return jsonify({"translation": translation})
    except Exception as e:
        logger.error("Translation error: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@app.post("/translate/attention")
def translate_with_attention():
    """Translate and return attention weights.

    Request body (JSON):
        {
            "text": "<source text>"
        }

    Response (JSON):
        {
            "translation": "<translated text>",
            "attention": [[...], [...], ...]   // 2D list (tgt_len, src_len) or null
        }
    """
    if not inference_service.is_ready():
        return jsonify({"error": "Model not ready. Check server logs."}), 503

    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "'text' field is required and must be non-empty."}), 400

    text = data["text"].strip()

    try:
        translation, attention = inference_service.translate_with_attention(text)
        attention_list = attention.tolist() if attention is not None else None
        return jsonify({"translation": translation, "attention": attention_list})
    except Exception as e:
        logger.error("Translation error: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _initialize_service()

    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("APP_PORT", 5000))
    debug = os.environ.get("APP_DEBUG", "false").lower() == "true"

    logger.info("Starting NMT app on %s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)
