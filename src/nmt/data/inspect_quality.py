"""Deep quality inspection of the English-Amharic parallel corpus.

Checks performed
----------------
1.  Basic counts — total pairs, missing values, empty strings
2.  Duplicate analysis — exact duplicates, duplicate sources, duplicate targets
3.  Length statistics — word and character counts for both languages
4.  Length distribution — bucketed histogram with bar chart
5.  Length ratio analysis — flags extreme length mismatches
6.  Script / encoding checks — detects non-Ethiopic Amharic, non-ASCII English noise
7.  Noise patterns — HTML tags, URLs, numbers-only, single-token sentences
8.  Longest & shortest examples — spot-check edge cases
9.  Most frequent sentences — detects boilerplate / near-duplicates
10. Summary scorecard — overall quality estimate

Usage
-----
    python src/nmt/data/inspect_quality.py
    python src/nmt/data/inspect_quality.py --file data/raw/mt560_amharic_english/train.parquet
    python src/nmt/data/inspect_quality.py --file data/raw/mt560_amharic_english/train.parquet --save-report reports/tables/quality_report.txt
"""

import argparse
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_FILE = Path("data/raw/mt560_amharic_english/train.parquet")
EN_COL = "eng"
AM_COL = "amh"

# Ethiopic Unicode block: U+1200–U+137F (basic), U+1380–U+139F (supplement)
ETHIOPIC_RE = re.compile(r"[\u1200-\u139F]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE      = re.compile(r"https?://\S+|www\.\S+")
DIGITS_ONLY = re.compile(r"^\s*[\d\s\.,]+\s*$")


# ── helpers ──────────────────────────────────────────────────────────────────

def _sep(title: str = "", width: int = 60) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def _bar(value: int, total: int, width: int = 30) -> str:
    filled = int(value / total * width) if total else 0
    return "█" * filled + "░" * (width - filled)


def _word_count(text: str) -> int:
    return len(text.split())


def _char_count(text: str) -> int:
    return len(text.strip())


def _has_ethiopic(text: str) -> bool:
    return bool(ETHIOPIC_RE.search(text))


def _has_html(text: str) -> bool:
    return bool(HTML_TAG_RE.search(text))


def _has_url(text: str) -> bool:
    return bool(URL_RE.search(text))


def _is_digits_only(text: str) -> bool:
    return bool(DIGITS_ONLY.match(text))


def _length_buckets(lengths: list, bins=None) -> dict:
    bins = bins or [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 9999)]
    buckets = {}
    for lo, hi in bins:
        label = f"{lo}-{hi}" if hi < 9999 else f"{lo}+"
        buckets[label] = sum(1 for l in lengths if lo <= l <= hi)
    return buckets


# ── main ─────────────────────────────────────────────────────────────────────

def inspect(file_path: Path, save_report: Path | None = None) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas is required:  pip install pandas pyarrow")
        sys.exit(1)

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        print("Run  python src/nmt/data/download.py  first.")
        sys.exit(1)

    # ── load ─────────────────────────────────────────────────────────────────
    df = pd.read_parquet(file_path)
    lines = []  # collect output lines for optional file save

    def out(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        print(text, **kwargs)
        lines.append(text)

    def sep(title=""):
        _sep(title)
        lines.append(f"\n── {title} " + "─" * max(0, 55 - len(title)))

    # ── detect columns ───────────────────────────────────────────────────────
    cols = df.columns.tolist()
    en_col = next(
        (c for c in cols if c.lower() in ("eng", "en", "english", "source", "src")), None
    )
    am_col = next(
        (c for c in cols if c.lower() in ("amh", "am", "amharic", "target", "tgt")), None
    )
    if not en_col or not am_col:
        print(f"[ERROR] Cannot detect English/Amharic columns in: {cols}")
        sys.exit(1)

    en = df[en_col].fillna("").astype(str).tolist()
    am = df[am_col].fillna("").astype(str).tolist()
    N  = len(en)

    out(f"\nFile       : {file_path}")
    out(f"EN column  : '{en_col}'  |  AM column: '{am_col}'")
    out(f"Total rows : {N:,}")

    # ══════════════════════════════════════════════════════════════════════
    # 1. Missing / empty
    # ══════════════════════════════════════════════════════════════════════
    sep("1 · MISSING & EMPTY VALUES")
    null_en    = df[en_col].isna().sum()
    null_am    = df[am_col].isna().sum()
    empty_en   = sum(1 for s in en if not s.strip())
    empty_am   = sum(1 for s in am if not s.strip())
    either_empty = sum(1 for a, b in zip(en, am) if not a.strip() or not b.strip())

    out(f"  NaN English           : {null_en:>8,}")
    out(f"  NaN Amharic           : {null_am:>8,}")
    out(f"  Empty English         : {empty_en:>8,}")
    out(f"  Empty Amharic         : {empty_am:>8,}")
    out(f"  Either side empty     : {either_empty:>8,}  ({either_empty/N*100:.2f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # 2. Duplicates
    # ══════════════════════════════════════════════════════════════════════
    sep("2 · DUPLICATES")
    pairs         = list(zip(en, am))
    pair_counts   = Counter(pairs)
    exact_dups    = sum(v - 1 for v in pair_counts.values() if v > 1)
    dup_src       = sum(v - 1 for v in Counter(en).values() if v > 1)
    dup_tgt       = sum(v - 1 for v in Counter(am).values() if v > 1)
    unique_pairs  = len(pair_counts)

    out(f"  Unique pairs          : {unique_pairs:>8,}  ({unique_pairs/N*100:.2f}%)")
    out(f"  Exact duplicate pairs : {exact_dups:>8,}")
    out(f"  Duplicate EN sources  : {dup_src:>8,}")
    out(f"  Duplicate AM targets  : {dup_tgt:>8,}")

    if exact_dups > 0:
        out("\n  Top 3 most repeated pairs:")
        for (e, a), cnt in pair_counts.most_common(3):
            out(f"    [{cnt}×]  EN: {e[:70]}")
            out(f"           AM: {a[:70]}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. Length statistics (words)
    # ══════════════════════════════════════════════════════════════════════
    sep("3 · LENGTH STATISTICS (words)")
    en_wlen = [_word_count(s) for s in en]
    am_wlen = [_word_count(s) for s in am]

    def _stats(lengths):
        n = len(lengths)
        s = sorted(lengths)
        return {
            "min":    s[0],
            "max":    s[-1],
            "mean":   sum(lengths) / n,
            "median": s[n // 2],
            "p95":    s[int(n * 0.95)],
            "p99":    s[int(n * 0.99)],
        }

    en_s = _stats(en_wlen)
    am_s = _stats(am_wlen)

    out(f"  {'Metric':<12}  {'English':>10}  {'Amharic':>10}")
    out(f"  {'------':<12}  {'-------':>10}  {'-------':>10}")
    for k in ("min", "max", "mean", "median", "p95", "p99"):
        out(f"  {k:<12}  {en_s[k]:>10.1f}  {am_s[k]:>10.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # 4. Length distribution histogram
    # ══════════════════════════════════════════════════════════════════════
    sep("4 · LENGTH DISTRIBUTION (words per sentence)")
    en_dist = _length_buckets(en_wlen)
    am_dist = _length_buckets(am_wlen)
    max_count = max(max(en_dist.values()), max(am_dist.values()), 1)

    out(f"  {'Bucket':>8}  {'English':>8}   {'':30}  {'Amharic':>8}")
    for bucket in en_dist:
        ec = en_dist[bucket]
        ac = am_dist.get(bucket, 0)
        out(
            f"  {bucket:>8}  {ec:>8,}  {_bar(ec, max_count)}  "
            f"  {ac:>8,}  {_bar(ac, max_count)}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # 5. Length ratio (EN words / AM words)
    # ══════════════════════════════════════════════════════════════════════
    sep("5 · LENGTH RATIO (EN words ÷ AM words)")
    ratios = [
        en_wlen[i] / max(am_wlen[i], 1) for i in range(N)
    ]
    very_short_tgt = sum(1 for r in ratios if r > 4.0)   # EN much longer than AM
    very_short_src = sum(1 for r in ratios if r < 0.25)  # AM much longer than EN
    balanced       = N - very_short_tgt - very_short_src

    out(f"  Balanced (0.25–4.0×)  : {balanced:>8,}  ({balanced/N*100:.2f}%)")
    out(f"  EN >> AM  (ratio >4×) : {very_short_tgt:>8,}  ({very_short_tgt/N*100:.2f}%)")
    out(f"  AM >> EN  (ratio <.25): {very_short_src:>8,}  ({very_short_src/N*100:.2f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # 6. Script / encoding checks
    # ══════════════════════════════════════════════════════════════════════
    sep("6 · SCRIPT & ENCODING CHECKS")
    no_ethiopic   = sum(1 for s in am if s.strip() and not _has_ethiopic(s))
    en_has_ethiopic = sum(1 for s in en if _has_ethiopic(s))

    out(f"  AM sentences with no Ethiopic script : {no_ethiopic:>8,}  ({no_ethiopic/N*100:.2f}%)")
    out(f"  EN sentences containing Ethiopic     : {en_has_ethiopic:>8,}  ({en_has_ethiopic/N*100:.2f}%)")

    if no_ethiopic > 0:
        out("\n  Sample AM sentences with no Ethiopic:")
        shown = 0
        for s in am:
            if s.strip() and not _has_ethiopic(s):
                out(f"    → {s[:100]}")
                shown += 1
                if shown >= 3:
                    break

    # ══════════════════════════════════════════════════════════════════════
    # 7. Noise patterns
    # ══════════════════════════════════════════════════════════════════════
    sep("7 · NOISE PATTERNS")
    en_html        = sum(1 for s in en if _has_html(s))
    am_html        = sum(1 for s in am if _has_html(s))
    en_url         = sum(1 for s in en if _has_url(s))
    am_url         = sum(1 for s in am if _has_url(s))
    en_digits      = sum(1 for s in en if _is_digits_only(s))
    am_digits      = sum(1 for s in am if _is_digits_only(s))
    en_single      = sum(1 for s in en if _word_count(s) == 1)
    am_single      = sum(1 for s in am if _word_count(s) == 1)
    en_long        = sum(1 for w in en_wlen if w > 100)
    am_long        = sum(1 for w in am_wlen if w > 100)

    out(f"  {'Pattern':<35}  {'English':>8}  {'Amharic':>8}")
    out(f"  {'-------':<35}  {'-------':>8}  {'-------':>8}")
    rows = [
        ("Contains HTML tags",        en_html,   am_html),
        ("Contains URLs",             en_url,    am_url),
        ("Digits / numbers only",     en_digits, am_digits),
        ("Single-token sentences",    en_single, am_single),
        ("Sentences > 100 words",     en_long,   am_long),
    ]
    for label, ev, av in rows:
        out(f"  {label:<35}  {ev:>8,}  {av:>8,}")

    # ══════════════════════════════════════════════════════════════════════
    # 8. Shortest & longest examples
    # ══════════════════════════════════════════════════════════════════════
    sep("8 · EDGE CASES — SHORTEST & LONGEST")
    indexed = sorted(enumerate(en_wlen), key=lambda x: x[1])

    out("\n  3 SHORTEST (by English word count):")
    for idx, wl in indexed[:3]:
        out(f"  [{wl} words]  EN: {en[idx][:80]}")
        out(f"              AM: {am[idx][:80]}")

    out("\n  3 LONGEST (by English word count):")
    for idx, wl in reversed(indexed[-3:]):
        out(f"  [{wl} words]  EN: {en[idx][:80]}…")
        out(f"              AM: {am[idx][:80]}…")

    # ══════════════════════════════════════════════════════════════════════
    # 9. Most frequent sentences
    # ══════════════════════════════════════════════════════════════════════
    sep("9 · MOST FREQUENT ENGLISH SENTENCES (top 5)")
    for sent, cnt in Counter(en).most_common(5):
        out(f"  [{cnt}×]  {sent[:90]}")

    # ══════════════════════════════════════════════════════════════════════
    # 10. Summary scorecard
    # ══════════════════════════════════════════════════════════════════════
    sep("10 · SUMMARY SCORECARD")
    total_flagged = (
        either_empty + exact_dups + no_ethiopic +
        en_html + am_html + en_url + am_url +
        en_digits + am_digits + very_short_tgt + very_short_src
    )
    # Clamp to N (same pair can be flagged multiple ways)
    total_flagged = min(total_flagged, N)
    clean_est = N - total_flagged

    out(f"  Total pairs              : {N:>8,}")
    out(f"  Flagged (any issue)      : {total_flagged:>8,}  ({total_flagged/N*100:.1f}%)")
    out(f"  Clean estimate           : {clean_est:>8,}  ({clean_est/N*100:.1f}%)")

    quality = clean_est / N * 100
    if quality >= 90:
        grade = "🟢  GOOD"
    elif quality >= 75:
        grade = "🟡  FAIR — some cleaning recommended"
    else:
        grade = "🔴  POOR — cleaning required before training"

    out(f"\n  Quality grade: {grade}")
    out()

    # ── optional save ────────────────────────────────────────────────────────
    if save_report:
        save_report = Path(save_report)
        save_report.parent.mkdir(parents=True, exist_ok=True)
        save_report.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nReport saved to: {save_report}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Deep quality inspection of the NMT corpus.")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help=f"Path to the parquet data file (default: {DEFAULT_FILE})",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path to save the report as a text file",
    )
    args = parser.parse_args()
    inspect(args.file, args.save_report)


if __name__ == "__main__":
    main()
