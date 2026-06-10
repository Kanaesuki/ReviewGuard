import argparse
from pathlib import Path
from typing import Dict

import joblib

from src.config import MODEL_PATH


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.train` first.")
    return joblib.load(model_path)


def predict_text(text: str, model_path: Path = MODEL_PATH) -> Dict[str, object]:
    model = load_model(model_path)
    label = int(model.predict([text])[0])
    result = "疑似虚假评论/水军评论" if label == 1 else "正常评论"

    score = None
    classifier = model.named_steps.get("classifier")
    if hasattr(classifier, "predict_proba"):
        score = float(model.predict_proba([text])[0][1])
    elif hasattr(classifier, "decision_function"):
        score = float(model.decision_function([text])[0])

    return {"label": label, "result": result, "score": score}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single review.")
    parser.add_argument("--text", required=True, help="Review text.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(predict_text(args.text))
