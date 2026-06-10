from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fake_review_detector.joblib"
REPORT_PATH = MODEL_DIR / "metrics_report.txt"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

TRAIN_CSV = DATA_DIR / "train.csv"
DEV_CSV = DATA_DIR / "dev.csv"
TEST_CSV = DATA_DIR / "test.csv"
STOPWORDS_PATH = DATA_DIR / "hit_stopwords.txt"
DATA_STATS_PATH = DOCS_DIR / "data_statistics.md"

# 京东数据集标签定义（全队统一）
LABEL_FAKE = 0
LABEL_REAL = 1
LABEL_NAMES = {
    LABEL_FAKE: "虚假/水军评论",
    LABEL_REAL: "真实用户评论",
}

# 内置基础停用词，会与 hit_stopwords.txt 合并
BASE_STOPWORDS = {
    "的",
    "了",
    "和",
    "是",
    "我",
    "也",
    "就",
    "都",
    "而",
    "及",
    "与",
    "着",
    "或",
    "一个",
    "没有",
}


def load_stopwords():
    """Load stopwords from the HIT list and merge with built-in words."""
    words = set(BASE_STOPWORDS)
    if STOPWORDS_PATH.exists():
        with STOPWORDS_PATH.open(encoding="utf-8") as file:
            for line in file:
                word = line.strip()
                if word:
                    words.add(word)
    return words
