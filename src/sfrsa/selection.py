from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .embeddings import bert_embeddings

SelectionMethod = Literal["kmeans", "cgc", "dpp", "iu-dpp", "iudpp"]


@dataclass(frozen=True)
class SelectionResult:
    """Detailed result returned when ``return_result=True``."""

    selected_indices: np.ndarray
    selected_texts: list[str]
    selected_data: pd.DataFrame | None
    relevance_scores: np.ndarray
    method: str


def select_augmented_texts(
    candidates: pd.DataFrame | Sequence[str],
    minority_texts: pd.DataFrame | Sequence[str] | None = None,
    *,
    method: SelectionMethod = "dpp",
    budget: int,
    text_column: str = "text",
    candidate_embeddings: np.ndarray | None = None,
    minority_embeddings: np.ndarray | None = None,
    embedding_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    embedding_model_name: str = "bert-base-uncased",
    embedding_max_length: int = 512,
    embedding_batch_size: int = 8,
    embedding_device: str | None = None,
    utility_scores: Sequence[float] | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    prefilter_top_m: int | None = None,
    random_state: int | None = 42,
    return_result: bool = False,
) -> pd.DataFrame | list[str] | SelectionResult:
    """Select a budgeted subset of generated text candidates.

    Parameters
    ----------
    candidates:
        Candidate generated texts, either a dataframe or a sequence of strings.
    minority_texts:
        Original minority-class texts used to define the relevance centroid.
        Required unless ``minority_embeddings`` are supplied. For IU-DPP this can
        be omitted because quality comes from ``utility_scores``.
    method:
        ``"kmeans"``/``"cgc"``, ``"dpp"``, or ``"iu-dpp"``.
    budget:
        Number of candidates to select.
    candidate_embeddings, minority_embeddings:
        Optional precomputed embeddings. Use these to reuse BERT or other
        encoder outputs.
    embedding_fn:
        Callable that maps text sequences to a 2-D numpy array. If omitted,
        BERT embeddings are computed with ``embedding_model_name``.
    embedding_model_name, embedding_max_length, embedding_batch_size, embedding_device:
        BERT embedding settings used when ``embedding_fn`` and precomputed
        embeddings are not supplied.
    utility_scores:
        Training-aware utility scores for IU-DPP, one per candidate.
    alpha, beta, gamma:
        Method hyperparameters. ``alpha`` controls DPP relevance quality,
        ``beta`` controls IU-DPP utility quality, and ``gamma`` controls the
        cosine diversity kernel.
    prefilter_top_m:
        Optional DPP speed/relevance prefilter. Must be at least ``budget``.
    return_result:
        Return a ``SelectionResult`` instead of selected dataframe/list.
    """

    candidate_texts, candidate_frame = _coerce_texts(candidates, text_column)
    minority_list, _ = _coerce_texts(minority_texts, text_column) if minority_texts is not None else (None, None)
    n = len(candidate_texts)
    _validate_budget(budget, n)

    normalized_method = method.lower()
    if normalized_method == "cgc":
        normalized_method = "kmeans"
    if normalized_method == "iudpp":
        normalized_method = "iu-dpp"
    if normalized_method not in {"kmeans", "dpp", "iu-dpp"}:
        raise ValueError("method must be one of: 'kmeans', 'dpp', 'iu-dpp'")
    if normalized_method == "iu-dpp" and utility_scores is None:
        raise ValueError("utility_scores are required for method='iu-dpp'")

    needs_minority = normalized_method in {"kmeans", "dpp"} or minority_embeddings is not None
    if needs_minority and minority_list is None and minority_embeddings is None:
        raise ValueError("minority_texts or minority_embeddings are required for this method")

    minority_for_embeddings = minority_list if needs_minority else None
    cand_emb, min_emb = _resolve_embeddings(
        candidate_texts,
        minority_for_embeddings,
        candidate_embeddings,
        minority_embeddings,
        embedding_fn,
        embedding_model_name,
        embedding_max_length,
        embedding_batch_size,
        embedding_device,
    )

    if normalized_method == "kmeans":
        selected, relevance = _select_kmeans(cand_emb, min_emb, budget, random_state)
    elif normalized_method == "dpp":
        selected, relevance = _select_dpp(
            cand_emb,
            min_emb,
            budget,
            alpha=alpha,
            gamma=gamma,
            prefilter_top_m=prefilter_top_m,
        )
    else:
        selected, relevance = _select_iu_dpp(
            cand_emb,
            utility_scores,
            budget,
            beta=beta,
            gamma=gamma,
            prefilter_top_m=prefilter_top_m,
        )

    selected_texts = [candidate_texts[i] for i in selected]
    selected_data = candidate_frame.iloc[selected].copy() if candidate_frame is not None else None

    if return_result:
        return SelectionResult(
            selected_indices=np.asarray(selected, dtype=int),
            selected_texts=selected_texts,
            selected_data=selected_data,
            relevance_scores=relevance,
            method=normalized_method,
        )
    return selected_data if selected_data is not None else selected_texts


def _coerce_texts(data: pd.DataFrame | Sequence[str], text_column: str) -> tuple[list[str], pd.DataFrame | None]:
    if isinstance(data, pd.DataFrame):
        if text_column not in data.columns:
            raise ValueError(f"text_column '{text_column}' is not in the dataframe")
        return data[text_column].astype(str).tolist(), data.reset_index(drop=True)
    return [str(x) for x in data], None


def _validate_budget(budget: int, n: int) -> None:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > n:
        raise ValueError(f"budget cannot exceed number of candidates ({n})")


def _resolve_embeddings(
    candidate_texts: list[str],
    minority_texts: list[str] | None,
    candidate_embeddings: np.ndarray | None,
    minority_embeddings: np.ndarray | None,
    embedding_fn: Callable[[Sequence[str]], np.ndarray] | None,
    embedding_model_name: str,
    embedding_max_length: int,
    embedding_batch_size: int,
    embedding_device: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if candidate_embeddings is not None:
        cand = np.asarray(candidate_embeddings, dtype=np.float64)
        if cand.ndim != 2 or cand.shape[0] != len(candidate_texts):
            raise ValueError("candidate_embeddings must have shape (len(candidates), d)")
        if minority_embeddings is not None:
            min_emb = np.asarray(minority_embeddings, dtype=np.float64)
        elif minority_texts is not None and embedding_fn is not None:
            min_emb = np.asarray(embedding_fn(minority_texts), dtype=np.float64)
        elif minority_texts is not None:
            raise ValueError(
                "minority_embeddings are required when candidate_embeddings are provided "
                "without an embedding_fn"
            )
        else:
            min_emb = None
        return cand, min_emb

    if embedding_fn is not None:
        cand = np.asarray(embedding_fn(candidate_texts), dtype=np.float64)
        min_emb = None if minority_texts is None else np.asarray(embedding_fn(minority_texts), dtype=np.float64)
        return cand, min_emb

    embedder = lambda texts: bert_embeddings(
        texts,
        model_name=embedding_model_name,
        max_length=embedding_max_length,
        batch_size=embedding_batch_size,
        device=embedding_device,
    )
    if minority_texts is None:
        cand = embedder(candidate_texts)
        return cand, None
    all_texts = candidate_texts + minority_texts
    all_emb = embedder(all_texts)
    return all_emb[: len(candidate_texts)], all_emb[len(candidate_texts) :]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _minority_centroid(minority_embeddings: np.ndarray) -> np.ndarray:
    if minority_embeddings is None or len(minority_embeddings) == 0:
        raise ValueError("minority embeddings are required")
    centroid = _l2_normalize(minority_embeddings).mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm <= 1e-12:
        raise ValueError("minority centroid has near-zero norm")
    return centroid / norm


def _relevance(candidate_embeddings: np.ndarray, minority_embeddings: np.ndarray) -> np.ndarray:
    cand = _l2_normalize(candidate_embeddings)
    centroid = _minority_centroid(minority_embeddings)
    return cand @ centroid


def _select_kmeans(
    candidate_embeddings: np.ndarray,
    minority_embeddings: np.ndarray,
    budget: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    relevance = _relevance(candidate_embeddings, minority_embeddings)
    labels = KMeans(n_clusters=budget, n_init=10, random_state=random_state).fit_predict(candidate_embeddings)
    selected: list[int] = []
    for cluster_id in range(budget):
        members = np.flatnonzero(labels == cluster_id)
        if len(members) == 0:
            continue
        selected.append(int(members[np.argmax(relevance[members])]))
    return np.asarray(selected, dtype=int), relevance


def _select_dpp(
    candidate_embeddings: np.ndarray,
    minority_embeddings: np.ndarray,
    budget: int,
    *,
    alpha: float,
    gamma: float,
    prefilter_top_m: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    relevance = _relevance(candidate_embeddings, minority_embeddings)
    quality = np.exp(alpha * (relevance - relevance.max()))
    selected = _quality_diversity_dpp(candidate_embeddings, quality, budget, gamma, prefilter_top_m, relevance)
    return selected, relevance


def _select_iu_dpp(
    candidate_embeddings: np.ndarray,
    utility_scores: Sequence[float],
    budget: int,
    *,
    beta: float,
    gamma: float,
    prefilter_top_m: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    utility = np.asarray(utility_scores, dtype=np.float64)
    if utility.shape != (candidate_embeddings.shape[0],):
        raise ValueError("utility_scores must contain one score per candidate")
    utility_norm = _robust_scale(utility)
    quality = np.exp(beta * utility_norm)
    selected = _quality_diversity_dpp(candidate_embeddings, quality, budget, gamma, prefilter_top_m, utility_norm)
    return selected, utility_norm


def _quality_diversity_dpp(
    candidate_embeddings: np.ndarray,
    quality: np.ndarray,
    budget: int,
    gamma: float,
    prefilter_top_m: int | None,
    prefilter_scores: np.ndarray,
) -> np.ndarray:
    indices = np.arange(candidate_embeddings.shape[0])
    if prefilter_top_m is not None:
        if prefilter_top_m < budget:
            raise ValueError("prefilter_top_m must be at least budget")
        top_m = min(prefilter_top_m, len(indices))
        indices = np.argsort(-prefilter_scores)[:top_m]
        candidate_embeddings = candidate_embeddings[indices]
        quality = quality[indices]

    emb = _l2_normalize(candidate_embeddings)
    cosine = emb @ emb.T
    kernel = np.exp(-gamma * (1.0 - cosine))
    l_ensemble = (quality[:, None] * kernel) * quality[None, :]
    selected_local = _dpp_greedy_map(l_ensemble, budget)
    return indices[selected_local]


def _dpp_greedy_map(l_ensemble: np.ndarray, budget: int, eps: float = 1e-12) -> np.ndarray:
    l_ensemble = np.asarray(l_ensemble, dtype=np.float64)
    n = l_ensemble.shape[0]
    if l_ensemble.shape != (n, n):
        raise ValueError("DPP L-ensemble must be square")

    cis = np.zeros((budget, n), dtype=np.float64)
    di2s = np.clip(np.diag(l_ensemble).copy(), 0.0, None)
    selected: list[int] = []
    selected_mask = np.zeros(n, dtype=bool)

    for step in range(budget):
        di2s[selected_mask] = -np.inf
        item = int(np.argmax(di2s))
        if not np.isfinite(di2s[item]) or di2s[item] <= eps:
            break

        selected.append(item)
        selected_mask[item] = True

        if step == budget - 1:
            break

        di = np.sqrt(max(di2s[item], eps))
        if step == 0:
            eis = l_ensemble[item, :] / di
        else:
            eis = (l_ensemble[item, :] - cis[:step, item] @ cis[:step, :]) / di
        cis[step, :] = eis
        di2s -= eis * eis
        di2s = np.maximum(di2s, 0.0)

    return np.asarray(selected, dtype=int)


def _robust_scale(scores: np.ndarray, clip: float = 5.0) -> np.ndarray:
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    scaled = (scores - median) / max(1.4826 * mad, 1e-12)
    return np.clip(scaled, -clip, clip)
