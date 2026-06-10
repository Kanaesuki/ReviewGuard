import argparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    CONFUSION_MATRIX_PATH,
    LABEL_COLUMN,
    MODEL_DIR,
    MODEL_PATH,
    REPORT_PATH,
    TEXT_COLUMN,
)
from src.preprocess import preprocess_text


def build_model(model_name: str) -> Pipeline:
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "naive_bayes": MultinomialNB(),
        "linear_svm": LinearSVC(class_weight="balanced"),
    }

    if model_name not in classifiers:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    tokenizer=str.split,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            ("classifier", classifiers[model_name]),
        ]
    )


def plot_confusion_matrix(y_true, y_pred) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "fake"],
        yticklabels=["normal", "fake"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=180)
    plt.close()


def train(data_path: str, model_name: str) -> str:
    data = pd.read_csv(data_path)
    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x_train, x_test, y_train, y_test = train_test_split(
        data[TEXT_COLUMN],
        data[LABEL_COLUMN],
        test_size=0.25,
        random_state=42,
        stratify=data[LABEL_COLUMN],
    )

    model = build_model(model_name)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["normal", "fake"],
        digits=4,
        zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")
    plot_confusion_matrix(y_test, y_pred)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fake review detector.")
    parser.add_argument("--data", default="data/sample_reviews.csv", help="CSV data path.")
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=["logistic_regression", "naive_bayes", "linear_svm"],
        help="Classifier name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(train(args.data, args.model))
