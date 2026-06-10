from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import (
    DATA_DIR,
    DEV_CSV,
    LABEL_COLUMN,
    RAW_DATA_DIR,
    TEST_CSV,
    TEXT_COLUMN,
    TRAIN_CSV,
)


def load_jd_txt(file_path: Path) -> pd.DataFrame:
    """Load JD raw review file with tab-separated label and text columns."""
    dataframe = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=[LABEL_COLUMN, TEXT_COLUMN],
        on_bad_lines="skip",
    )
    dataframe[TEXT_COLUMN] = dataframe[TEXT_COLUMN].astype(str)
    dataframe[LABEL_COLUMN] = dataframe[LABEL_COLUMN].astype(int)
    return dataframe


def save_dataset_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe[[LABEL_COLUMN, TEXT_COLUMN]].to_csv(output_path, index=False, encoding="utf-8")


def convert_raw_to_csv(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = DATA_DIR,
) -> Dict[str, Path]:
    """Convert train/dev/test txt files into CSV datasets."""
    mapping = {
        "train": raw_dir / "train.txt",
        "dev": raw_dir / "dev.txt",
        "test": raw_dir / "test.txt",
    }
    outputs: Dict[str, Path] = {}

    for name, source_path in mapping.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Missing raw dataset: {source_path}")

        dataframe = load_jd_txt(source_path)
        output_path = output_dir / f"{name}.csv"
        save_dataset_csv(dataframe, output_path)
        outputs[name] = output_path

    return outputs


def load_dataset_splits(
    train_path: Path = TRAIN_CSV,
    dev_path: Path = DEV_CSV,
    test_path: Path = TEST_CSV,
) -> Dict[str, pd.DataFrame]:
    """Load train/dev/test CSV files."""
    splits = {
        "train": train_path,
        "dev": dev_path,
        "test": test_path,
    }

    datasets: Dict[str, pd.DataFrame] = {}
    for name, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset split: {path}")
        datasets[name] = pd.read_csv(path)
        missing = {TEXT_COLUMN, LABEL_COLUMN} - set(datasets[name].columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    return datasets
