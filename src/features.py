"""Biểu diễn văn bản, feature selection và handcrafted features."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

HANDCRAFTED_FEATURE_NAMES = [
    "word_count",
    "first_person_ratio",
    "negation_ratio",
    "absolutist_ratio",
    "sentence_count",
    "exclamation_ratio",
    "warning_word_count",
]

FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "myself"}
NEGATION_WORDS = {"not", "no", "never", "nothing", "none", "neither", "nor"}
ABSOLUTIST_WORDS = {
    "always",
    "never",
    "everyone",
    "everybody",
    "nobody",
    "everything",
    "nothing",
    "completely",
    "totally",
    "entirely",
    "absolutely",
}
WARNING_WORDS = {
    "suicide",
    "suicidal",
    "kill",
    "die",
    "death",
    "worthless",
    "hopeless",
}

TOKEN_PATTERN = re.compile(r"\b[a-z]+(?:'[a-z]+)?\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class TfidfConfig:
    """Cấu hình feature selection gián tiếp của TF-IDF.

    - ``max_features`` giới hạn số đặc trưng có độ ưu tiên cao nhất.
    - ``min_df`` loại token quá hiếm.
    - ``max_df`` loại token xuất hiện quá phổ biến.
    - ``ngram_range`` giữ unigram và bigram.
    """

    max_features: int = 3000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.9
    sublinear_tf: bool = True


def build_tfidf_vectorizer(config: TfidfConfig | None = None) -> TfidfVectorizer:
    cfg = config or TfidfConfig()
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        dtype=np.float32,
    )


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]


def extract_handcrafted_features(text: str) -> dict[str, float]:
    """Trích xuất 7 đặc trưng thủ công từ văn bản gần với dữ liệu gốc.

    Regex có ranh giới từ được dùng để tránh lỗi đếm ``die`` trong một từ dài
    hơn hoặc bỏ sót đại từ đi kèm dấu câu.
    """
    raw_text = "" if pd.isna(text) else str(text)
    tokens = _tokens(raw_text)
    n_words = max(len(tokens), 1)

    # Các contraction như don't/can't vẫn được tokenizer giữ nguyên. Đếm chúng
    # như tín hiệu phủ định bên cạnh các từ phủ định độc lập.
    contraction_negations = sum(token.endswith("n't") for token in tokens)

    return {
        "word_count": float(len(tokens)),
        "first_person_ratio": sum(token in FIRST_PERSON_WORDS for token in tokens) / n_words,
        "negation_ratio": (
            sum(token in NEGATION_WORDS for token in tokens) + contraction_negations
        )
        / n_words,
        "absolutist_ratio": sum(token in ABSOLUTIST_WORDS for token in tokens) / n_words,
        "sentence_count": float(len(re.findall(r"[.!?]+", raw_text))),
        "exclamation_ratio": raw_text.count("!") / max(len(raw_text), 1),
        "warning_word_count": float(sum(token in WARNING_WORDS for token in tokens)),
    }


def handcrafted_matrix(texts: Sequence[str] | Iterable[str]) -> np.ndarray:
    rows = [extract_handcrafted_features(text) for text in texts]
    return pd.DataFrame(rows, columns=HANDCRAFTED_FEATURE_NAMES).to_numpy(dtype=np.float32)


def fit_transform_xgb_features(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    train_raw_texts: Sequence[str],
    test_raw_texts: Sequence[str],
    *,
    tfidf_config: TfidfConfig | None = None,
    n_components: int = 300,
    seed: int = 42,
):
    """Fit TF-IDF và SVD chỉ trên train, sau đó ghép handcrafted features.

    Trả về ``X_train``, ``X_test``, vectorizer, SVD và tên đặc trưng.
    """
    vectorizer = build_tfidf_vectorizer(tfidf_config)
    x_train_tfidf = vectorizer.fit_transform(train_texts)
    x_test_tfidf = vectorizer.transform(test_texts)

    max_valid_components = max(1, min(x_train_tfidf.shape[0] - 1, x_train_tfidf.shape[1] - 1))
    actual_components = min(n_components, max_valid_components)

    svd = TruncatedSVD(n_components=actual_components, random_state=seed)
    x_train_svd = svd.fit_transform(x_train_tfidf).astype(np.float32)
    x_test_svd = svd.transform(x_test_tfidf).astype(np.float32)

    x_train_hc = handcrafted_matrix(train_raw_texts)
    x_test_hc = handcrafted_matrix(test_raw_texts)

    x_train = np.hstack([x_train_svd, x_train_hc]).astype(np.float32)
    x_test = np.hstack([x_test_svd, x_test_hc]).astype(np.float32)
    feature_names = [f"svd_component_{idx + 1:03d}" for idx in range(actual_components)]
    feature_names.extend(HANDCRAFTED_FEATURE_NAMES)

    return x_train, x_test, vectorizer, svd, feature_names
