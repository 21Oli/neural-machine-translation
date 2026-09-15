"""Data loading, cleaning, and splitting utilities."""

from .audit import audit_dataset
from .clean import clean_corpus
from .download import download_dataset
from .normalize import normalize_text
from .split import split_dataset

__all__ = [
    "download_dataset",
    "audit_dataset",
    "clean_corpus",
    "normalize_text",
    "split_dataset",
]

# NMTDataset requires torch — import lazily to avoid hard dependency
def __getattr__(name):
    if name == "NMTDataset":
        from .dataset import NMTDataset
        return NMTDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
