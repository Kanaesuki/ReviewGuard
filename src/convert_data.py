import argparse

from src.data_io import convert_raw_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw JD txt datasets into train/dev/test CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    outputs = convert_raw_to_csv()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
