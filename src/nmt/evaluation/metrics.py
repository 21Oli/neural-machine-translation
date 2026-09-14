"""Translation quality metrics (BLEU, chrF, TER via sacrebleu)."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    tokenize: str = "13a",
) -> Dict[str, float]:
    """Compute sentence-level and corpus BLEU using sacrebleu.

    Args:
        hypotheses: List of model output strings.
        references: List of reference strings (one per hypothesis).
        tokenize: sacrebleu tokenizer name ('13a', 'intl', 'char', 'none').

    Returns:
        Dictionary with 'bleu', 'bleu_1', ..., 'bleu_4', 'bp'.
    """
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError("Install sacrebleu: pip install sacrebleu") from e

    result = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)
    return {
        "bleu": result.score,
        "bleu_1": result.precisions[0],
        "bleu_2": result.precisions[1],
        "bleu_3": result.precisions[2],
        "bleu_4": result.precisions[3],
        "bp": result.bp,
        "ratio": result.sys_len / result.ref_len if result.ref_len > 0 else 0.0,
    }


def compute_corpus_bleu(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Return scalar corpus BLEU score.

    Args:
        hypotheses: Model output strings.
        references: Reference strings.

    Returns:
        Corpus BLEU score (0–100 scale).
    """
    metrics = compute_bleu(hypotheses, references)
    logger.info("Corpus BLEU: %.2f", metrics["bleu"])
    return metrics["bleu"]


def compute_chrf(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Compute chrF score.

    Args:
        hypotheses: Model output strings.
        references: Reference strings.

    Returns:
        chrF score (0–100 scale).
    """
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError("Install sacrebleu: pip install sacrebleu") from e

    result = sacrebleu.corpus_chrf(hypotheses, [references])
    logger.info("chrF: %.2f", result.score)
    return result.score


def compute_ter(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Compute Translation Edit Rate (TER).

    Args:
        hypotheses: Model output strings.
        references: Reference strings.

    Returns:
        TER score (lower is better).
    """
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError("Install sacrebleu: pip install sacrebleu") from e

    result = sacrebleu.corpus_ter(hypotheses, [references])
    logger.info("TER: %.2f", result.score)
    return result.score
