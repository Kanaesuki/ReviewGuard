import html
import re
from functools import lru_cache
from typing import Iterable, List

try:
    import jieba
except ModuleNotFoundError:
    jieba = None

from src.config import load_stopwords


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#\d+;")
NON_CHINESE_PATTERN = re.compile(r"[^\u4e00-\u9fa5]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    return frozenset(load_stopwords())


def clean_text(text: str) -> str:
    """Normalize noisy review text before tokenization."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = HTML_ENTITY_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = NON_CHINESE_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize(text: str, *, min_word_len: int = 2) -> List[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    stopwords = get_stopwords()
    if jieba is not None:
        words = jieba.lcut(cleaned)
    else:
        words = re.findall(r"[\u4e00-\u9fa5]+", cleaned)

    return [
        word
        for word in words
        if word.strip()
        and len(word) >= min_word_len
        and word not in stopwords
    ]


def preprocess_text(text: str) -> str:
    return " ".join(tokenize(text))


def preprocess_many(texts: Iterable[str]) -> List[str]:
    return [preprocess_text(text) for text in texts]
