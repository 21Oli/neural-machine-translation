"""Dataset download utilities."""

import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASET_ID = "michsethowusu/english-amharic_sentence-pairs_mt560"
OUTPUT_DIR = Path("data/raw/mt560_amharic_english")


def download_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    output_dir: str = "data/raw",
    cache_dir: Optional[str] = None,
) -> Path:
    """Download a dataset from HuggingFace Hub and save it to disk.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. 'masakhane/mafand').
        subset: Optional dataset subset/config name.
        output_dir: Local directory to save raw data.
        cache_dir: Optional HuggingFace cache directory.

    Returns:
        Path to the downloaded dataset directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset '%s' (subset=%s)...", dataset_name, subset)

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if subset:
        kwargs["name"] = subset

    dataset = load_dataset(dataset_name, **kwargs)
    save_path = output_path / dataset_name.replace("/", "_")
    dataset.save_to_disk(str(save_path))

    logger.info("Dataset saved to %s", save_path)
    return save_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET_ID}")

    dataset = load_dataset(DATASET_ID)

    print("\nDataset downloaded successfully.")
    print(dataset)

    train_data = dataset["train"]

    print(f"\nNumber of rows: {len(train_data)}")
    print(f"Columns: {train_data.column_names}")

    # Save the train split as parquet
    output_file = OUTPUT_DIR / "train.parquet"
    train_data.to_parquet(str(output_file))

    print(f"\nSaved dataset to: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")

    # Display a few examples for an initial structural check
    print("\nFirst three examples:")
    for example in train_data.select(range(min(3, len(train_data)))):
        print(example)


if __name__ == "__main__":
    main()
