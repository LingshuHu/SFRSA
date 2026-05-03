from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class Step0UtilityModel:
    """Step-0 classifier and validation-gradient state for IU-DPP utility."""

    model_type: str
    positive_label: Any
    negative_label: Any
    validation_gradient: np.ndarray
    text_column: str
    label_column: str
    pipeline: Pipeline | None = None
    bert_model: Any | None = None
    tokenizer: Any | None = None
    device: str = "cpu"
    max_length: int = 100
    batch_size: int = 8


def train_step0_utility_model(
    training_data: pd.DataFrame,
    *,
    model_type: Literal["logistic", "bert"] = "logistic",
    text_column: str = "text",
    label_column: str = "label",
    positive_label: Any = 1,
    negative_label: Any = 0,
    validation_size: float | int = 0.1,
    random_state: int | None = 42,
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    max_iter: int = 1000,
    bert_model_name: str = "bert-base-uncased",
    bert_max_length: int = 100,
    bert_batch_size: int = 8,
    bert_epochs: int = 3,
    bert_learning_rate: float = 2e-5,
    device: str = "cpu",
) -> Step0UtilityModel:
    """Train the Step-0 classifier used to compute IU-DPP utility scores.

    ``model_type="logistic"`` trains a lightweight class-weighted logistic
    classifier. ``model_type="bert"`` fine-tunes a BERT sequence classifier and
    uses classifier-head gradients, matching the influence-utility calculation
    used by IU-DPP. For a candidate ``x_i`` with minority label, utility is the
    dot product between the validation-loss gradient and the candidate-loss
    gradient.
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

    normalized_model_type = model_type.lower()
    if normalized_model_type == "logistic":
        return _train_logistic_step0(
            x_train,
            x_val,
            y_train,
            y_val,
            positive_label=positive_label,
            negative_label=negative_label,
            text_column=text_column,
            label_column=label_column,
            random_state=random_state,
            max_features=max_features,
            ngram_range=ngram_range,
            max_iter=max_iter,
        )
    if normalized_model_type == "bert":
        return _train_bert_step0(
            x_train,
            x_val,
            y_train,
            y_val,
            positive_label=positive_label,
            negative_label=negative_label,
            text_column=text_column,
            label_column=label_column,
            model_name=bert_model_name,
            max_length=bert_max_length,
            batch_size=bert_batch_size,
            epochs=bert_epochs,
            learning_rate=bert_learning_rate,
            device=device,
            random_state=random_state,
        )
    raise ValueError("model_type must be either 'logistic' or 'bert'")


def _train_logistic_step0(
    x_train: Sequence[str],
    x_val: Sequence[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    positive_label: Any,
    negative_label: Any,
    text_column: str,
    label_column: str,
    random_state: int | None,
    max_features: int,
    ngram_range: tuple[int, int],
    max_iter: int,
) -> Step0UtilityModel:
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
        model_type="logistic",
        pipeline=pipeline,
        positive_label=positive_label,
        negative_label=negative_label,
        validation_gradient=validation_gradient,
        text_column=text_column,
        label_column=label_column,
    )


def _train_bert_step0(
    x_train: Sequence[str],
    x_val: Sequence[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    positive_label: Any,
    negative_label: Any,
    text_column: str,
    label_column: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: str,
    random_state: int | None,
) -> Step0UtilityModel:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError(
            "BERT Step-0 training requires torch and transformers. "
            "Install the package dependencies and rerun with model_type='bert'."
        ) from exc

    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    bert_model.to(device)

    train_ds = _bert_dataset(tokenizer, x_train, y_train, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=learning_rate)
    counts = np.bincount(y_train, minlength=2)
    weights = len(y_train) / (2.0 * np.maximum(counts, 1))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    bert_model.train()
    for _ in range(epochs):
        for ids, mask, labels in train_loader:
            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = bert_model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    validation_gradient = _bert_mean_loss_gradient(
        bert_model,
        tokenizer,
        x_val,
        y_val,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    return Step0UtilityModel(
        model_type="bert",
        bert_model=bert_model,
        tokenizer=tokenizer,
        positive_label=positive_label,
        negative_label=negative_label,
        validation_gradient=validation_gradient,
        text_column=text_column,
        label_column=label_column,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
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
        if model.model_type == "logistic":
            if model.pipeline is None:
                raise ValueError("Step0UtilityModel is missing its logistic pipeline")
            grad = _mean_loss_gradient(model.pipeline, [text], np.asarray([y_value]))
        elif model.model_type == "bert":
            grad = _bert_mean_loss_gradient(
                model.bert_model,
                model.tokenizer,
                [text],
                np.asarray([y_value]),
                max_length=model.max_length,
                batch_size=model.batch_size,
                device=model.device,
            )
        else:
            raise ValueError(f"unsupported Step-0 model_type: {model.model_type}")
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


def _bert_dataset(tokenizer: Any, texts: Sequence[str], y: np.ndarray, *, max_length: int):
    import torch
    from torch.utils.data import TensorDataset

    encoded = tokenizer(
        [str(text) for text in texts],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(y, dtype=torch.long)
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)


def _bert_head_parameters(bert_model: Any) -> list[Any]:
    params = [param for name, param in bert_model.named_parameters() if "classifier" in name and param.requires_grad]
    if not params:
        params = [param for param in bert_model.parameters() if param.requires_grad]
    return params


def _flatten_torch_grads(params: Sequence[Any]) -> np.ndarray:
    import torch

    grads = []
    for param in params:
        grad = param.grad
        if grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(grad.detach().reshape(-1).cpu())
    if not grads:
        return np.asarray([], dtype=np.float32)
    return torch.cat(grads).numpy().astype(np.float32)


def _bert_mean_loss_gradient(
    bert_model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    y: np.ndarray,
    *,
    max_length: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError(
            "BERT utility scoring requires torch and transformers. "
            "Install the package dependencies and rerun."
        ) from exc

    if bert_model is None or tokenizer is None:
        raise ValueError("BERT Step0UtilityModel is missing its model or tokenizer")

    dataset = _bert_dataset(tokenizer, texts, y, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    params = _bert_head_parameters(bert_model)
    loss_fn = torch.nn.CrossEntropyLoss()
    grad_sum = None
    batches = 0

    bert_model.eval()
    for ids, mask, labels in loader:
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        bert_model.zero_grad(set_to_none=True)
        logits = bert_model(input_ids=ids, attention_mask=mask).logits
        loss = loss_fn(logits, labels)
        loss.backward()
        grad = _flatten_torch_grads(params)
        grad_sum = grad if grad_sum is None else grad_sum + grad
        batches += 1

    if grad_sum is None:
        return np.asarray([], dtype=np.float32)
    return (grad_sum / max(batches, 1)).astype(np.float32)
