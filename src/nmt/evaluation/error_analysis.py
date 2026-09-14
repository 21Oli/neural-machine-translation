"""Error analysis tools — categorizes translation errors and attention patterns."""

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Analyzes translation errors between hypotheses and references.

    Args:
        src_sentences: Source sentences.
        hypotheses: Model translation hypotheses.
        references: Gold reference translations.
    """

    def __init__(
        self,
        src_sentences: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> None:
        assert len(src_sentences) == len(hypotheses) == len(references)
        self.src_sentences = src_sentences
        self.hypotheses = hypotheses
        self.references = references

    def find_exact_matches(self) -> List[int]:
        """Return indices of sentences where hypothesis == reference."""
        return [i for i, (h, r) in enumerate(zip(self.hypotheses, self.references)) if h.strip() == r.strip()]

    def find_empty_hypotheses(self) -> List[int]:
        """Return indices of sentences with empty model output."""
        return [i for i, h in enumerate(self.hypotheses) if not h.strip()]

    def length_ratio_stats(self) -> Dict[str, float]:
        """Compute hypothesis/reference length ratio statistics."""
        ratios = [
            len(h.split()) / max(len(r.split()), 1)
            for h, r in zip(self.hypotheses, self.references)
        ]
        return {
            "mean_ratio": sum(ratios) / len(ratios),
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "over_generation": sum(1 for r in ratios if r > 1.2) / len(ratios),
            "under_generation": sum(1 for r in ratios if r < 0.8) / len(ratios),
        }

    def most_common_errors(self, top_k: int = 20) -> List[Tuple[str, int]]:
        """Find the most frequently mistranslated source words.

        Compares hypothesis words to reference words on a per-sentence basis
        and counts words that appear in the reference but not the hypothesis.

        Args:
            top_k: Number of top error tokens to return.

        Returns:
            List of (token, count) tuples.
        """
        missed: Counter = Counter()
        for hyp, ref in zip(self.hypotheses, self.references):
            hyp_tokens = set(hyp.lower().split())
            ref_tokens = set(ref.lower().split())
            missed.update(ref_tokens - hyp_tokens)
        return missed.most_common(top_k)

    def sentence_bleu_distribution(self) -> Dict[str, int]:
        """Bucket per-sentence BLEU scores into ranges."""
        try:
            from sacrebleu import sentence_bleu
        except ImportError as e:
            raise ImportError("Install sacrebleu: pip install sacrebleu") from e

        buckets = {"0-10": 0, "10-25": 0, "25-50": 0, "50-75": 0, "75-100": 0}
        for hyp, ref in zip(self.hypotheses, self.references):
            score = sentence_bleu(hyp, [ref]).score
            if score < 10:
                buckets["0-10"] += 1
            elif score < 25:
                buckets["10-25"] += 1
            elif score < 50:
                buckets["25-50"] += 1
            elif score < 75:
                buckets["50-75"] += 1
            else:
                buckets["75-100"] += 1
        return buckets

    def get_worst_translations(self, n: int = 10) -> List[Dict]:
        """Return the n worst translations by sentence BLEU score.

        Args:
            n: Number of worst examples to return.

        Returns:
            List of dicts with 'src', 'hypothesis', 'reference', 'bleu' keys.
        """
        try:
            from sacrebleu import sentence_bleu
        except ImportError as e:
            raise ImportError("Install sacrebleu: pip install sacrebleu") from e

        scored = []
        for src, hyp, ref in zip(self.src_sentences, self.hypotheses, self.references):
            score = sentence_bleu(hyp, [ref]).score
            scored.append({"src": src, "hypothesis": hyp, "reference": ref, "bleu": score})

        return sorted(scored, key=lambda x: x["bleu"])[:n]
