
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


DEFAULT_INPUT_FILE = (
    Path("data")
    / "interim"
    / "mt560_amharic_english"
    / "train_clean.parquet"
)

DEFAULT_OUTPUT_DIR = (
    Path("data")
    / "processed"
    / "mt560_amharic_english"
)

SOURCE_COLUMN = "eng"
TARGET_COLUMN = "amh"


def split_parallel_corpus(
    input_file: str | Path = DEFAULT_INPUT_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> Dict[str, int]:
    """
    Split a cleaned English–Amharic parallel corpus into train,
    validation, and test subsets.

    The split is performed at the row level, preserving the alignment
    between the English source and Amharic target sentences.
    """

    if not abs(
        train_ratio + validation_ratio + test_ratio - 1.0
    ) < 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")

    for name, ratio in {
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
    }.items():
        if ratio <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset was not found: {input_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    required_columns = {SOURCE_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df[[SOURCE_COLUMN, TARGET_COLUMN]].copy()

    if df.empty:
        raise ValueError("The input dataset is empty.")

    if df[[SOURCE_COLUMN, TARGET_COLUMN]].isna().any().any():
        raise ValueError("The input dataset contains missing values.")

    if df.duplicated(subset=[SOURCE_COLUMN, TARGET_COLUMN]).any():
        raise ValueError(
            "The input dataset contains duplicate translation pairs."
        )

    # Shuffle the complete paired rows.
    shuffled_df = df.sample(
        frac=1.0,
        random_state=random_seed,
    ).reset_index(drop=True)

    total_rows = len(shuffled_df)

    train_end = int(total_rows * train_ratio)
    validation_end = train_end + int(
        total_rows * validation_ratio
    )

    train_df = shuffled_df.iloc[:train_end].copy()
    validation_df = shuffled_df.iloc[
        train_end:validation_end
    ].copy()
    test_df = shuffled_df.iloc[validation_end:].copy()

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    split_counts: Dict[str, int] = {}

    for split_name, split_df in splits.items():
        parquet_file = output_path / f"{split_name}.parquet"
        english_file = output_path / f"{split_name}.en"
        amharic_file = output_path / f"{split_name}.am"

        split_df.to_parquet(
            parquet_file,
            index=False,
        )

        english_file.write_text(
            "\n".join(split_df[SOURCE_COLUMN].tolist()),
            encoding="utf-8",
        )

        amharic_file.write_text(
            "\n".join(split_df[TARGET_COLUMN].tolist()),
            encoding="utf-8",
        )

        split_counts[split_name] = len(split_df)

    # Verify that all rows were preserved.
    if sum(split_counts.values()) != total_rows:
        raise RuntimeError(
            "Split sizes do not add up to the original dataset size."
        )

    # Verify that no exact translation pair overlaps between splits.
    train_pairs = set(
        zip(train_df[SOURCE_COLUMN], train_df[TARGET_COLUMN])
    )
    validation_pairs = set(
        zip(
            validation_df[SOURCE_COLUMN],
            validation_df[TARGET_COLUMN],
        )
    )
    test_pairs = set(
        zip(test_df[SOURCE_COLUMN], test_df[TARGET_COLUMN])
    )

    if train_pairs & validation_pairs:
        raise RuntimeError(
            "Data leakage detected between train and validation sets."
        )

    if train_pairs & test_pairs:
        raise RuntimeError(
            "Data leakage detected between train and test sets."
        )

    if validation_pairs & test_pairs:
        raise RuntimeError(
            "Data leakage detected between validation and test sets."
        )

    return split_counts


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Split the cleaned English–Amharic parallel corpus "
            "into train, validation, and test datasets."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the cleaned input Parquet file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where split files will be saved.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Training split ratio.",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Test split ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the dataset split from the command line."""

    args = parse_arguments()

    counts = split_parallel_corpus(
        input_file=args.input_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    total = sum(counts.values())

    print("=" * 60)
    print("TRAIN / VALIDATION / TEST SPLIT COMPLETE")
    print("=" * 60)
    print(f"Input file : {args.input_file}")
    print(f"Output dir : {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    print(f"Train      : {counts['train']:,} pairs")
    print(f"Validation : {counts['validation']:,} pairs")
    print(f"Test       : {counts['test']:,} pairs")
    print(f"Total      : {total:,} pairs")
    print()
    print("Saved files:")
    for split_name in ("train", "validation", "test"):
        print(f"  - {args.output_dir / f'{split_name}.parquet'}")
        print(f"  - {args.output_dir / f'{split_name}.en'}")
        print(f"  - {args.output_dir / f'{split_name}.am'}")
    print()
    print("Alignment and leakage checks passed.")


if __name__ == "__main__":
    main()