"""Evaluation: BLEU metrics, benchmarking, and error analysis."""

from .benchmark import benchmark_model
from .error_analysis import ErrorAnalyzer
from .metrics import compute_bleu, compute_corpus_bleu

__all__ = ["compute_bleu", "compute_corpus_bleu", "benchmark_model", "ErrorAnalyzer"]
