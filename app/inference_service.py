"""Inference service — loads the trained model once and exposes a translate() function.

This module is the single point of model loading for the Flask app. It reads
paths and settings from environment variables (via .env) so the app itself
stays configuration-free.
"""

import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Module-level singletons — populated by initialize()
_translator = None
_src_tokenizer = None
_tgt_tokenizer = None


def initialize(
    model_path: Optional[str] = None,
    src_tokenizer_path: Optional[str] = None,
    tgt_tokenizer_path: Optional[str] = None,
    device_str: Optional[str] = None,
) -> None:
    """Load model and tokenizers into memory.

    Falls back to environment variables when arguments are not supplied:
        NMT_MODEL_PATH, NMT_SRC_TOKENIZER, NMT_TGT_TOKENIZER, DEVICE

    Args:
        model_path: Path to a saved .pt model checkpoint.
        src_tokenizer_path: Path to the source SentencePiece .model file.
        tgt_tokenizer_path: Path to the target SentencePiece .model file.
        device_str: Torch device string ('cpu', 'cuda', 'mps').
    """
    global _translator

    model_path = model_path or os.environ.get("NMT_MODEL_PATH")
    src_tokenizer_path = src_tokenizer_path or os.environ.get("NMT_SRC_TOKENIZER")
    tgt_tokenizer_path = tgt_tokenizer_path or os.environ.get("NMT_TGT_TOKENIZER")
    device_str = device_str or os.environ.get("DEVICE", "cpu")

    if not all([model_path, src_tokenizer_path, tgt_tokenizer_path]):
        raise EnvironmentError(
            "Model and tokenizer paths must be provided via arguments or "
            "NMT_MODEL_PATH / NMT_SRC_TOKENIZER / NMT_TGT_TOKENIZER environment variables."
        )

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from nmt.inference.preprocessing import InferencePreprocessor
    from nmt.inference.translator import Translator
    from nmt.tokenization.tokenizer import NMTTokenizer
    from nmt.training.checkpoints import CheckpointManager

    device = torch.device(device_str)
    logger.info("Initializing inference service on device '%s'...", device_str)

    # Load tokenizers
    src_tok = NMTTokenizer(model_path=src_tokenizer_path)
    tgt_tok = NMTTokenizer(model_path=tgt_tokenizer_path)

    # Load model architecture + weights from checkpoint
    # The checkpoint must contain a 'model_state_dict' and optionally 'model_config'
    checkpoint = torch.load(model_path, map_location="cpu")
    model_config = checkpoint.get("model_config", {})
    model_type = model_config.get("type", "attention_seq2seq")

    model = _build_model(model_type, model_config, src_tok.vocab_size, tgt_tok.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    preprocessor = InferencePreprocessor(
        src_lang=model_config.get("src_lang", "en"),
        remove_html=True,
        remove_url=True,
    )

    _translator = Translator(
        model=model,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        device=device,
        preprocessor=preprocessor,
        max_len=model_config.get("max_len", 100),
    )
    logger.info("Inference service ready.")


def translate(text: str) -> str:
    """Translate a single text string.

    Args:
        text: Raw source language input.

    Returns:
        Translated target language string.

    Raises:
        RuntimeError: If the service has not been initialized.
    """
    if _translator is None:
        raise RuntimeError("Inference service is not initialized. Call initialize() first.")
    return _translator.translate(text)


def translate_with_attention(text: str) -> Tuple[str, Optional[torch.Tensor]]:
    """Translate and return attention weights alongside the translation.

    Args:
        text: Raw source language input.

    Returns:
        (translation, attention_tensor) — attention may be None for non-attention models.
    """
    if _translator is None:
        raise RuntimeError("Inference service is not initialized. Call initialize() first.")
    return _translator.translate_with_attention(text)


def is_ready() -> bool:
    """Return True if the service has been successfully initialized."""
    return _translator is not None


def _build_model(model_type: str, config: dict, src_vocab_size: int, tgt_vocab_size: int):
    """Instantiate the correct model architecture from a config dict."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from nmt.models.encoder import Encoder

    embed_dim = config.get("embed_dim", 256)
    hidden_dim = config.get("hidden_dim", 512)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.0)
    bidirectional = config.get("bidirectional_encoder", True)

    encoder = Encoder(
        vocab_size=src_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    if model_type == "attention_seq2seq":
        from nmt.models.attention_seq2seq import AttentionDecoder, AttentionSeq2Seq

        encoder_dim = hidden_dim * 2 if bidirectional else hidden_dim
        decoder = AttentionDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
            attention_dim=config.get("attention_dim", 256),
            num_layers=num_layers,
            dropout=dropout,
        )
        return AttentionSeq2Seq(encoder, decoder)
    else:
        from nmt.models.decoder import Decoder
        from nmt.models.seq2seq import Seq2Seq

        decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        return Seq2Seq(encoder, decoder)
