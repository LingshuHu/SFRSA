from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class Step0UtilityModel:
    """Step-0 classifier and validation-gradient state for IU-DPP utility."""

    pipeline: Pipeline
    positive_label: Any
    negative_label: Any
    validation_gradient: np.ndarray
    text_column: str
    label_column: str


def train_step0_utility_model(
    training_data: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "label",
    positive_label: Any = 1,
    negative_label: Any = 0,
    validation_size: float | int = 0.1,
    random_state: int | None = 42,
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    max_iter: int = 1000,
) -> Step0UtilityModel:
    """Train the Step-0 classifier used to compute IU-DPP utility scores.

    This implements the first-order utility idea from SFRSA with a lightweight
    class-weighted logistic classifier. For a candidate ``x_i`` with minority
    label, utility is the dot product between the validation-loss gradient and
    the candidate-loss gradient. Higher values indicate candidates expected to
    reduce validation loss more strongly after a small training step.
    """

    _validate_columns(training_data, text_column, label_column)
    df = training_data[training_data[label_column].isin([positive_label, negative_label])].copy()
    if df.empty:
        raise ValueError("training_data has no rows with the requested labels")

    texts = df[text_column].astype(str)
    labels = (df[label_column] == positive_label).astype(int).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError("training_data must contain both positive and negative labels")

    stratify = labels if _can_stratify(labels, validation_size) else None
    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    validation_gradient = _mean_loss_gradient(pipeline, x_val, y_val)
    return Step0UtilityModel(
        pipeline=pipeline,
        positive_label=positive_label,
        negative_label=negative_label,
        validation_gradient=validation_gradient,
        text_column=text_column,
        label_column=label_column,
    )


def compute_utility_scores(
    model: Step0UtilityModel,
    candidates: pd.DataFrame | Sequence[str],
    *,
    text_column: str | None = None,
    candidate_label: Any | None = None,
) -> np.ndarray:
    """Compute first-order IU-DPP utility scores for candidate texts."""

    if isinstance(candidates, pd.DataFrame):
        column = text_column or model.text_column
        if column not in candidates.columns:
            raise ValueError(f"text_column '{column}' is not in candidates")
        texts = candidates[column].astype(str)
    else:
        texts = pd.Series([str(x) for x in candidates])

    label = model.positive_label if candidate_label is None else candidate_label
    y_value = 1 if label == model.positive_label else 0
    utilities = []
    for text in texts:
        grad = _mean_loss_gradient(model.pipeline, [text], np.asarray([y_value]))
        utilities.append(float(np.dot(model.validation_gradient, grad)))
    return np.asarray(utilities, dtype=np.float64)


def _validate_columns(data: pd.DataFrame, text_column: str, label_column: str) -> None:
    if text_column not in data.columns:
        raise ValueError(f"text_column '{text_column}' is not in training_data")
    if label_column not in data.columns:
        raise ValueError(f"label_column '{label_column}' is not in training_data")


def _can_stratify(labels: np.ndarray, validation_size: float | int) -> bool:
    counts = np.bincount(labels)
    if np.any(counts < 2):
        return False
    if isinstance(validation_size, float):
        val_count = int(np.ceil(len(labels) * validation_size))
    else:
        val_count = int(validation_size)
    return val_count >= len(counts)


def _mean_loss_gradient(pipeline: Pipeline, texts: Sequence[str], y: np.ndarray) -> np.ndarray:
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    x = vectorizer.transform(texts)
    coef = classifier.coef_.reshape(-1)
    intercept = classifier.intercept_[0]
    logits = x @ coef + intercept
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits).reshape(-1)))
    residual = probs - y
    grad_w = np.asarray(x.multiply(residual[:, None]).mean(axis=0)).reshape(-1)
    grad_b = np.asarray([residual.mean()])
    return np.concatenate([grad_w, grad_b])
