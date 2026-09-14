"""Model benchmarking — evaluates a model on a test set and saves results."""

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .metrics import compute_bleu, compute_chrf

logger = logging.getLogger(__name__)


def benchmark_model(
    model: nn.Module,
    src_sentences: List[str],
    ref_sentences: List[str],
    translate_fn: Callable[[nn.Module, str], str],
    output_dir: Optional[str] = None,
    model_name: str = "model",
) -> Dict:
    """Run full benchmark evaluation on a test set.

    Args:
        model: Trained model.
        src_sentences: Source sentences to translate.
        ref_sentences: Corresponding reference translations.
        translate_fn: Callable(model, src_text) → hypothesis string.
        output_dir: If given, saves results JSON and predictions to this directory.
        model_name: Label used in filenames.

    Returns:
        Dictionary of evaluation results.
    """
    model.eval()
    hypotheses: List[str] = []

    logger.info("Benchmarking '%s' on %d sentences...", model_name, len(src_sentences))
    start = time.time()

    for src in src_sentences:
        hyp = translate_fn(model, src)
        hypotheses.append(hyp)

    elapsed = time.time() - start
    sentences_per_second = len(src_sentences) / elapsed if elapsed > 0 else 0

    bleu_metrics = compute_bleu(hypotheses, ref_sentences)
    chrf_score = compute_chrf(hypotheses, ref_sentences)

    results = {
        "model": model_name,
        "num_sentences": len(src_sentences),
        "elapsed_seconds": round(elapsed, 2),
        "sentences_per_second": round(sentences_per_second, 2),
        **bleu_metrics,
        "chrf": chrf_score,
    }

    logger.info(
        "BLEU: %.2f | chrF: %.2f | Speed: %.1f sent/s",
        bleu_metrics["bleu"], chrf_score, sentences_per_second,
    )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_path = out / f"{model_name}_results.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        preds_path = out / f"{model_name}_predictions.txt"
        preds_path.write_text("\n".join(hypotheses), encoding="utf-8")
        logger.info("Results saved to %s", results_path)

    return results
