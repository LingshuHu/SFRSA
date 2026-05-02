from __future__ import annotations

from typing import Sequence

import numpy as np


def bert_text_preparation(text: str, tokenizer):
    """Prepare one text for BERT, following the original SFRSA notebooks."""

    marked_text = "[CLS] " + str(text) + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are missing
        raise ImportError("BERT embeddings require torch. Install with: pip install sfrsa") from exc

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model) -> list[float]:
    """Return the average token embedding from the sum of BERT's last four layers."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are missing
        raise ImportError("BERT embeddings require torch. Install with: pip install sfrsa") from exc

    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        hidden_states = outputs.hidden_states[-4:]

    token_embeddings = hidden_states[0] + hidden_states[1] + hidden_states[2] + hidden_states[3]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    token_embeddings = torch.mean(token_embeddings, dim=0)
    return token_embeddings.detach().cpu().tolist()


def bert_embeddings(
    texts: Sequence[str],
    *,
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
    batch_size: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """Embed texts with BERT using the SFRSA last-four-layers mean strategy.

    The original notebooks embed each text by summing BERT's last four hidden
    layers and averaging over tokens. This batched implementation preserves that
    representation while using tokenizer padding/truncation for package use.
    """

    try:
        import torch
        from transformers import BertModel, BertTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "BERT embeddings require torch and transformers. "
            "Install package dependencies, then rerun the selection."
        ) from exc

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    texts = [str(text) for text in texts]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            hidden_states = outputs.hidden_states[-4:]

        summed = hidden_states[0] + hidden_states[1] + hidden_states[2] + hidden_states[3]
        mask = encoded["attention_mask"].unsqueeze(-1).type_as(summed)
        pooled = (summed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        embeddings.append(pooled.detach().cpu().numpy())

    if not embeddings:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(embeddings).astype(np.float64)
