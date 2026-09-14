"""Dataset auditing — statistics, quality checks, and diagnostics."""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Path to the downloaded parquet file (relative to project root)
DATA_FILE = Path("data/raw/mt560_amharic_english/train.parquet")

# Column names in the michsethowusu/english-amharic dataset
EN_COL = "english"
AM_COL = "amharic"


def audit_dataset(
    src_sentences: List[str],
    tgt_sentences: List[str],
    src_lang: str = "src",
    tgt_lang: str = "tgt",
) -> Dict:
    """Compute statistics and quality metrics for a parallel corpus.

    Args:
        src_sentences: List of source language sentences.
        tgt_sentences: List of target language sentences.
        src_lang: Source language label for reporting.
        tgt_lang: Target language label for reporting.

    Returns:
        Dictionary of audit statistics.
    """
    assert len(src_sentences) == len(tgt_sentences), (
        "Source and target corpora must have the same number of sentences."
    )

    n = len(src_sentences)
    src_lengths = [len(s.split()) for s in src_sentences]
    tgt_lengths = [len(s.split()) for s in tgt_sentences]

    stats = {
        "num_pairs": n,
        f"{src_lang}_avg_len": round(sum(src_lengths) / n, 2) if n else 0,
        f"{tgt_lang}_avg_len": round(sum(tgt_lengths) / n, 2) if n else 0,
        f"{src_lang}_max_len": max(src_lengths) if src_lengths else 0,
        f"{tgt_lang}_max_len": max(tgt_lengths) if tgt_lengths else 0,
        f"{src_lang}_min_len": min(src_lengths) if src_lengths else 0,
        f"{tgt_lang}_min_len": min(tgt_lengths) if tgt_lengths else 0,
        "empty_src": sum(1 for s in src_sentences if not s.strip()),
        "empty_tgt": sum(1 for s in tgt_sentences if not s.strip()),
        "duplicate_pairs": _count_duplicates(src_sentences, tgt_sentences),
    }

    logger.info("Audit results: %s", stats)
    return stats


def _count_duplicates(src: List[str], tgt: List[str]) -> int:
    """Return the number of extra occurrences of duplicate sentence pairs."""
    pairs = list(zip(src, tgt))
    counts = Counter(pairs)
    return sum(v - 1 for v in counts.values() if v > 1)


def length_distribution(sentences: List[str]) -> Dict[str, int]:
    """Bucket sentences by word-count ranges.

    Args:
        sentences: List of sentences.

    Returns:
        Dictionary mapping length bucket labels to counts.
    """
    buckets: Dict[str, int] = {
        "1-10": 0, "11-20": 0, "21-50": 0, "51-100": 0, "100+": 0
    }
    for s in sentences:
        n = len(s.split())
        if n <= 10:
            buckets["1-10"] += 1
        elif n <= 20:
            buckets["11-20"] += 1
        elif n <= 50:
            buckets["21-50"] += 1
        elif n <= 100:
            buckets["51-100"] += 1
        else:
            buckets["100+"] += 1
    return buckets


def _print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    if not DATA_FILE.exists():
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        print("Run  python src/nmt/data/download.py  first.")
        return

    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas is required. Run: pip install pandas pyarrow")
        return

    print(f"Loading data from: {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)

    # Detect column names automatically if they differ
    cols = df.columns.tolist()
    print(f"Columns found: {cols}")

    # Try to auto-detect English and Amharic columns
    en_col = next((c for c in cols if "english" in c.lower() or c.lower() in ("en", "eng")), None)
    am_col = next((c for c in cols if "amharic" in c.lower() or c.lower() in ("am", "amh")), None)

    if not en_col or not am_col:
        print(f"[ERROR] Could not identify English/Amharic columns in: {cols}")
        print("Edit EN_COL / AM_COL at the top of audit.py to match your column names.")
        return

    print(f"Using columns  →  English: '{en_col}'  |  Amharic: '{am_col}'")

    en_sentences = df[en_col].fillna("").astype(str).tolist()
    am_sentences = df[am_col].fillna("").astype(str).tolist()

    # ------------------------------------------------------------------ #
    # 2. Core statistics
    # ------------------------------------------------------------------ #
    _print_section("CORE STATISTICS")
    stats = audit_dataset(en_sentences, am_sentences, src_lang="en", tgt_lang="am")
    col_w = 28
    for key, val in stats.items():
        print(f"  {key:<{col_w}} {val:>10,}" if isinstance(val, int) else f"  {key:<{col_w}} {val:>10.2f}")

    # ------------------------------------------------------------------ #
    # 3. Sentence length distributions
    # ------------------------------------------------------------------ #
    _print_section("ENGLISH — LENGTH DISTRIBUTION (words)")
    en_dist = length_distribution(en_sentences)
    for bucket, count in en_dist.items():
        bar = "█" * (count * 30 // max(en_dist.values(), default=1))
        print(f"  {bucket:>10}  {count:>8,}  {bar}")

    _print_section("AMHARIC — LENGTH DISTRIBUTION (words)")
    am_dist = length_distribution(am_sentences)
    for bucket, count in am_dist.items():
        bar = "█" * (count * 30 // max(am_dist.values(), default=1))
        print(f"  {bucket:>10}  {count:>8,}  {bar}")

    # ------------------------------------------------------------------ #
    # 4. Sample pairs
    # ------------------------------------------------------------------ #
    _print_section("SAMPLE SENTENCE PAIRS (first 5)")
    for i in range(min(5, len(df))):
        print(f"\n  [{i+1}] EN: {en_sentences[i]}")
        print(f"       AM: {am_sentences[i]}")

    # ------------------------------------------------------------------ #
    # 5. Quality flags
    # ------------------------------------------------------------------ #
    _print_section("QUALITY FLAGS")
    empty_en  = sum(1 for s in en_sentences if not s.strip())
    empty_am  = sum(1 for s in am_sentences if not s.strip())
    too_long  = sum(1 for s in en_sentences if len(s.split()) > 100)
    too_short = sum(1 for s in en_sentences if len(s.split()) < 2)
    dups      = stats["duplicate_pairs"]

    print(f"  {'Empty English sentences':<30} {empty_en:>8,}")
    print(f"  {'Empty Amharic sentences':<30} {empty_am:>8,}")
    print(f"  {'English sentences > 100 words':<30} {too_long:>8,}")
    print(f"  {'English sentences < 2 words':<30} {too_short:>8,}")
    print(f"  {'Duplicate pairs':<30} {dups:>8,}")

    total = len(en_sentences)
    flagged = empty_en + empty_am + too_long + too_short + dups
    print(f"\n  Total pairs     : {total:,}")
    print(f"  Flagged / noisy : {flagged:,}  ({flagged/total*100:.1f}%)")
    print(f"  Clean estimate  : {total - flagged:,}  ({(total-flagged)/total*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
