import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    DATA_STATS_PATH,
    FIGURES_DIR,
    LABEL_COLUMN,
    LABEL_FAKE,
    LABEL_NAMES,
    LABEL_REAL,
    TEXT_COLUMN,
)
from src.data_io import load_dataset_splits
from src.preprocess import preprocess_text


def _configure_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _label_counts(dataframe: pd.DataFrame) -> pd.Series:
    counts = dataframe[LABEL_COLUMN].value_counts().sort_index()
    return counts.rename(index=LABEL_NAMES)


def _format_length_summary(summary: pd.DataFrame) -> str:
    lines = [
        "| 类别 | 指标 | 样本数 | 均值 | 中位数 | 最小值 | 最大值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label_name in summary.index:
        for metric in ["char_len", "token_len"]:
            metric_label = "字符长度" if metric == "char_len" else "词数长度"
            row = summary.loc[label_name, metric]
            lines.append(
                f"| {label_name} | {metric_label} | {int(row['count'])} | "
                f"{row['mean']} | {row['median']} | {row['min']} | {row['max']} |"
            )
    return "\n".join(lines)


def _length_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    working = dataframe.copy()
    working["char_len"] = working[TEXT_COLUMN].astype(str).str.len()
    working["token_len"] = working[TEXT_COLUMN].astype(str).map(
        lambda text: len(preprocess_text(text).split())
    )
    working["label_name"] = working[LABEL_COLUMN].map(LABEL_NAMES)

    summary = (
        working.groupby("label_name")[["char_len", "token_len"]]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )
    return summary


def _plot_label_distribution(datasets: dict[str, pd.DataFrame], output_path: Path) -> None:
    rows = []
    for split_name, dataframe in datasets.items():
        counts = _label_counts(dataframe)
        for label, count in counts.items():
            rows.append({"数据集": split_name, "类别": label, "数量": count})

    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="数据集", y="数量", hue="类别")
    plt.title("各数据集类别分布")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_length_distribution(datasets: dict[str, pd.DataFrame], output_path: Path) -> None:
    frames = []
    for split_name, dataframe in datasets.items():
        part = dataframe[[LABEL_COLUMN, TEXT_COLUMN]].copy()
        part["数据集"] = split_name
        part["字符长度"] = part[TEXT_COLUMN].astype(str).str.len()
        part["类别"] = part[LABEL_COLUMN].map(LABEL_NAMES)
        frames.append(part)

    plot_df = pd.concat(frames, ignore_index=True)
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=plot_df,
        x="字符长度",
        hue="类别",
        bins=30,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("评论字符长度分布（全量数据）")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def _sample_examples(dataframe: pd.DataFrame, random_state: int = 42) -> dict[str, dict[str, str]]:
    fake_row = dataframe[dataframe[LABEL_COLUMN] == LABEL_FAKE].sample(1, random_state=random_state).iloc[0]
    real_row = dataframe[dataframe[LABEL_COLUMN] == LABEL_REAL].sample(1, random_state=random_state).iloc[0]

    return {
        "fake": {
            "raw": str(fake_row[TEXT_COLUMN]),
            "clean": preprocess_text(str(fake_row[TEXT_COLUMN])),
        },
        "real": {
            "raw": str(real_row[TEXT_COLUMN]),
            "clean": preprocess_text(str(real_row[TEXT_COLUMN])),
        },
    }


def build_markdown_report(datasets: dict[str, pd.DataFrame]) -> str:
    all_df = pd.concat(datasets.values(), ignore_index=True)
    total_counts = _label_counts(all_df)
    examples = _sample_examples(datasets["train"])

    lines = [
        "# 数据集统计报告",
        "",
        "## 1. 数据来源与标签定义",
        "",
        "- 数据来源：京东电商真实/虚假评论数据集",
        "- 标签 0：虚假/水军评论",
        "- 标签 1：真实用户评论",
        "",
        "## 2. 数据规模",
        "",
        "| 数据集 | 样本数 |",
        "| --- | ---: |",
    ]

    for split_name, dataframe in datasets.items():
        lines.append(f"| {split_name} | {len(dataframe)} |")
    lines.append(f"| 合计 | {len(all_df)} |")

    lines.extend(
        [
            "",
            "## 3. 类别分布",
            "",
            "### 全量分布",
            "",
            "| 类别 | 数量 | 占比 |",
            "| --- | ---: | ---: |",
        ]
    )

    for label_value, label_name in LABEL_NAMES.items():
        count = int(total_counts.get(label_name, 0))
        ratio = count / len(all_df) * 100
        lines.append(f"| {label_name} ({label_value}) | {count} | {ratio:.2f}% |")

    for split_name, dataframe in datasets.items():
        lines.extend(["", f"### {split_name}", ""])
        split_counts = _label_counts(dataframe)
        for label_name, count in split_counts.items():
            ratio = count / len(dataframe) * 100
            lines.append(f"- {label_name}：{count} 条（{ratio:.2f}%）")

    lines.extend(
        [
            "",
            "## 4. 文本长度统计",
            "",
            "字符长度基于原始评论，词数长度基于清洗+分词+去停用词后的结果。",
            "",
            _format_length_summary(_length_summary(all_df)),
            "",
            "## 5. 预处理样例",
            "",
            "### 虚假评论样例",
            "",
            f"- 清洗前：{examples['fake']['raw']}",
            f"- 清洗后：{examples['fake']['clean']}",
            "",
            "### 真实评论样例",
            "",
            f"- 清洗前：{examples['real']['raw']}",
            f"- 清洗后：{examples['real']['clean']}",
            "",
            "## 6. 图表文件",
            "",
            "- `docs/figures/label_distribution.png`",
            "- `docs/figures/text_length_distribution.png`",
        ]
    )

    return "\n".join(lines) + "\n"


def run_stats(output_md: Path = DATA_STATS_PATH) -> str:
    _configure_plot_style()
    datasets = load_dataset_splits()

    label_plot = FIGURES_DIR / "label_distribution.png"
    length_plot = FIGURES_DIR / "text_length_distribution.png"
    _plot_label_distribution(datasets, label_plot)
    _plot_length_distribution(datasets, length_plot)

    report = build_markdown_report(datasets)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset statistics and report.")
    parser.add_argument(
        "--output",
        default=str(DATA_STATS_PATH),
        help="Markdown report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_stats(Path(args.output))
    print(report)


if __name__ == "__main__":
    main()
