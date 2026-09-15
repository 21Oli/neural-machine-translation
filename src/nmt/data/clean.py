
"""
Corpus cleaning and normalization utilities for the English-Amharic NMT corpus.

When run directly:
    python src/nmt/data/clean.py

Reads:
    data/raw/mt560_amharic_english/train.parquet

Writes:
    data/interim/mt560_amharic_english/train_clean.parquet
    data/interim/mt560_amharic_english/train_clean.en
    data/interim/mt560_amharic_english/train_clean.am

Cleaning policy
---------------
1. Remove obvious HTML/markup tags.
2. Normalize whitespace.
3. Remove empty source-target pairs.
4. Remove pairs containing URLs.
5. Remove pairs outside the 2-100 word range.
6. Require Ethiopic script in the Amharic target.
7. Remove only exact duplicate translation pairs.
8. Preserve legitimate linguistic content and punctuation.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and columns
# ---------------------------------------------------------------------------

RAW_FILE = Path("data/raw/mt560_amharic_english/train.parquet")
INTERIM_DIR = Path("data/interim/mt560_amharic_english")

EN_COL = "eng"
AM_COL = "amh"


# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------

# Ethiopic Unicode block.
# Used as a basic integrity check for Amharic target sentences.
ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")


# Conservative HTML/markup tag detection.
#
# We intentionally avoid a broad pattern such as <[^>]+> because the corpus
# may contain legitimate angle-bracket content that is not HTML.
HTML_TAG_RE = re.compile(
    r"</?(?:p|br|div|span|a|b|i|strong|em|table|tr|td|li)"
    r"(?:\s[^>]*)?>",
    re.IGNORECASE,
)


# URL detection.
#
# URLs are excluded from this project because they are not central to the
# English-Amharic translation objective and can introduce unnecessary
# vocabulary and tokenization complexity.
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def remove_html_tags(text: str) -> str:
    """Remove obvious HTML/markup tags while preserving normal text."""
    return HTML_TAG_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip surrounding whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """
    Apply safe, conservative text normalization.

    URLs are not removed here because URL filtering is handled separately
    at the pair level so that it can be reported explicitly.
    """
    text = remove_html_tags(text)
    text = normalize_whitespace(text)
    return text


def _preprocess(text: str, lang: str) -> str:
    """
    Preprocess one sentence.

    The lang argument is retained for future language-specific normalization.
    """
    del lang
    return normalize_text(text)


# ---------------------------------------------------------------------------
# Corpus cleaning
# ---------------------------------------------------------------------------

def clean_corpus(
    src_sentences: List[str],
    tgt_sentences: List[str],
    min_len: int = 2,
    max_len: int = 100,
    remove_duplicates: bool = True,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Clean aligned source-target sentence pairs.

    Parameters
    ----------
    src_sentences:
        English source sentences.

    tgt_sentences:
        Amharic target sentences.

    min_len:
        Minimum allowed number of whitespace-separated words.

    max_len:
        Maximum allowed number of whitespace-separated words.

    remove_duplicates:
        Remove exact duplicate (English, Amharic) pairs when True.

    Returns
    -------
    src_clean:
        Clean English sentences.

    tgt_clean:
        Clean Amharic sentences.

    report:
        Dictionary containing counts for every cleaning step.
    """

    if len(src_sentences) != len(tgt_sentences):
        raise ValueError(
            "Source and target lists must have equal length."
        )

    original_count = len(src_sentences)

    # -----------------------------------------------------------------------
    # Step 1: Pair and normalize
    # -----------------------------------------------------------------------

    pairs = list(zip(src_sentences, tgt_sentences))

    pairs = [
        (
            _preprocess(source, "en"),
            _preprocess(target, "am"),
        )
        for source, target in pairs
    ]

    # -----------------------------------------------------------------------
    # Step 2: Remove empty pairs
    # -----------------------------------------------------------------------

    before = len(pairs)

    pairs = [
        (source, target)
        for source, target in pairs
        if source and target
    ]

    removed_empty = before - len(pairs)

    # -----------------------------------------------------------------------
    # Step 3: Remove URL-containing pairs
    # -----------------------------------------------------------------------

    before = len(pairs)

    pairs = [
        (source, target)
        for source, target in pairs
        if not URL_RE.search(source)
        and not URL_RE.search(target)
    ]

    removed_urls = before - len(pairs)

    # -----------------------------------------------------------------------
    # Step 4: Length filtering
    # -----------------------------------------------------------------------

    before = len(pairs)

    pairs = [
        (source, target)
        for source, target in pairs
        if (
            min_len <= len(source.split()) <= max_len
            and min_len <= len(target.split()) <= max_len
        )
    ]

    removed_length = before - len(pairs)

    # -----------------------------------------------------------------------
    # Step 5: Ethiopic-script integrity check
    # -----------------------------------------------------------------------

    before = len(pairs)

    pairs = [
        (source, target)
        for source, target in pairs
        if ETHIOPIC_RE.search(target)
    ]

    removed_script = before - len(pairs)

    # -----------------------------------------------------------------------
    # Step 6: Remove exact duplicate pairs
    # -----------------------------------------------------------------------

    removed_dups = 0

    if remove_duplicates:
        seen = set()
        deduped = []

        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                deduped.append(pair)

        removed_dups = len(pairs) - len(deduped)
        pairs = deduped

    # -----------------------------------------------------------------------
    # Step 7: Separate source and target sentences
    # -----------------------------------------------------------------------

    if pairs:
        src_clean, tgt_clean = zip(*pairs)
        src_clean = list(src_clean)
        tgt_clean = list(tgt_clean)
    else:
        src_clean = []
        tgt_clean = []

    # -----------------------------------------------------------------------
    # Cleaning report
    # -----------------------------------------------------------------------

    report = {
        "original": original_count,
        "removed_empty": removed_empty,
        "removed_urls": removed_urls,
        "removed_length": removed_length,
        "removed_script": removed_script,
        "removed_dups": removed_dups,
        "final": len(src_clean),
    }

    logger.info(
        "Cleaning complete: %d → %d pairs "
        "(empty=%d urls=%d length=%d script=%d dups=%d)",
        original_count,
        len(src_clean),
        removed_empty,
        removed_urls,
        removed_length,
        removed_script,
        removed_dups,
    )

    return src_clean, tgt_clean, report


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------

def _sep(title: str = "", width: int = 62) -> None:
    """Print a formatted console separator."""

    if title:
        pad = max(1, (width - len(title) - 2) // 2)
        remaining = width - pad - len(title) - 2

        print(
            f"\n{'─' * pad} "
            f"{title} "
            f"{'─' * max(1, remaining)}"
        )
    else:
        print("─" * width)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete corpus cleaning pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # -----------------------------------------------------------------------
    # Check pandas
    # -----------------------------------------------------------------------

    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas is required: pip install pandas pyarrow")
        return

    # -----------------------------------------------------------------------
    # Check input
    # -----------------------------------------------------------------------

    if not RAW_FILE.exists():
        print(f"[ERROR] Raw file not found: {RAW_FILE}")
        print("Run python src/nmt/data/download.py first.")
        return

    # -----------------------------------------------------------------------
    # Load raw corpus
    # -----------------------------------------------------------------------

    print(f"Loading : {RAW_FILE}")

    df = pd.read_parquet(RAW_FILE)

    print(
        f"Rows    : {len(df):,} | "
        f"Columns: {df.columns.tolist()}"
    )

    # Validate expected columns.
    missing_columns = {
        EN_COL,
        AM_COL,
    } - set(df.columns)

    if missing_columns:
        print(
            f"[ERROR] Missing required columns: "
            f"{sorted(missing_columns)}"
        )
        return

    en_raw = (
        df[EN_COL]
        .fillna("")
        .astype(str)
        .tolist()
    )

    am_raw = (
        df[AM_COL]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # -----------------------------------------------------------------------
    # Clean corpus
    # -----------------------------------------------------------------------

    _sep("CLEANING")

    en_clean, am_clean, report = clean_corpus(
        en_raw,
        am_raw,
        min_len=2,
        max_len=100,
        remove_duplicates=True,
    )

    # -----------------------------------------------------------------------
    # Cleaning report
    # -----------------------------------------------------------------------

    _sep("CLEANING REPORT")

    width = 38

    print(
        f"  {'Original pairs':<{width}} "
        f"{report['original']:>8,}"
    )

    print(
        f"  {'Removed — empty':<{width}} "
        f"{report['removed_empty']:>8,}"
    )

    print(
        f"  {'Removed — URLs':<{width}} "
        f"{report['removed_urls']:>8,}"
    )

    print(
        f"  {'Removed — length filter':<{width}} "
        f"{report['removed_length']:>8,}"
    )

    print(
        f"  {'Removed — no Ethiopic script':<{width}} "
        f"{report['removed_script']:>8,}"
    )

    print(
        f"  {'Removed — exact duplicates':<{width}} "
        f"{report['removed_dups']:>8,}"
    )

    print(
        f"  {'─' * (width + 12)}"
    )

    total_removed = report["original"] - report["final"]

    removal_pct = (
        total_removed / report["original"] * 100
        if report["original"]
        else 0.0
    )

    final_pct = (
        report["final"] / report["original"] * 100
        if report["original"]
        else 0.0
    )

    print(
        f"  {'Total removed':<{width}} "
        f"{total_removed:>8,} ({removal_pct:.1f}%)"
    )

    print(
        f"  {'Final clean pairs':<{width}} "
        f"{report['final']:>8,} ({final_pct:.1f}%)"
    )

    # -----------------------------------------------------------------------
    # Length statistics
    # -----------------------------------------------------------------------

    _sep("LENGTH STATS AFTER CLEANING (words)")

    if en_clean and am_clean:
        en_lens = [len(sentence.split()) for sentence in en_clean]
        am_lens = [len(sentence.split()) for sentence in am_clean]

        for lang, lengths in (
            ("English", en_lens),
            ("Amharic", am_lens),
        ):
            sorted_lengths = sorted(lengths)
            n = len(sorted_lengths)

            p95_index = min(
                n - 1,
                int(n * 0.95),
            )

            print(
                f"  {lang:<10} "
                f"min={sorted_lengths[0]} "
                f"max={sorted_lengths[-1]} "
                f"mean={sum(lengths) / n:.1f} "
                f"median={sorted_lengths[n // 2]} "
                f"p95={sorted_lengths[p95_index]}"
            )
    else:
        print("  No clean pairs remain.")

    # -----------------------------------------------------------------------
    # Sample pairs
    # -----------------------------------------------------------------------

    _sep("SAMPLE CLEAN PAIRS (first 5)")

    for i in range(min(5, len(en_clean))):
        print(f"\n  [{i + 1}] EN: {en_clean[i]}")
        print(f"       AM: {am_clean[i]}")

    # -----------------------------------------------------------------------
    # Save cleaned corpus
    # -----------------------------------------------------------------------

    _sep("SAVING")

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_df = pd.DataFrame(
        {
            EN_COL: en_clean,
            AM_COL: am_clean,
        }
    )

    # Save Parquet.
    parquet_path = INTERIM_DIR / "train_clean.parquet"

    out_df.to_parquet(
        parquet_path,
        index=False,
    )

    print(
        f"  Parquet : {parquet_path} "
        f"({parquet_path.stat().st_size / 1024**2:.2f} MB)"
    )

    # Save English text.
    en_txt = INTERIM_DIR / "train_clean.en"

    en_txt.write_text(
        "\n".join(en_clean),
        encoding="utf-8",
    )

    print(f"  English : {en_txt}")

    # Save Amharic text.
    am_txt = INTERIM_DIR / "train_clean.am"

    am_txt.write_text(
        "\n".join(am_clean),
        encoding="utf-8",
    )

    print(f"  Amharic : {am_txt}")

    # -----------------------------------------------------------------------
    # Completion
    # -----------------------------------------------------------------------

    print("\nDone.")


if __name__ == "__main__":
    main()
