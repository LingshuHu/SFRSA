from __future__ import annotations

import math
import re
from typing import Any, Sequence

import pandas as pd


DEFAULT_PROMPT = """Write {n} text(s) that are similar to the positive examples while different from the negative examples.

Positive examples:
{positive_examples}

Negative examples:
{negative_examples}
"""


def generate_few_random_shot(
    training_data: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "label",
    positive_label: Any = 1,
    negative_label: Any = 0,
    total: int,
    positive_examples: int = 10,
    negative_examples: int = 10,
    n_per_prompt: int = 1,
    prompt: str | None = None,
    model: str = "gpt-4.1-mini",
    client: Any | None = None,
    temperature: float = 0.9,
    random_state: int | None = None,
    max_attempts: int = 3,
) -> list[str]:
    """Generate texts with few-random-shot positive/negative examples.

    ``prompt`` may use ``{n}``, ``{positive_examples}``, and
    ``{negative_examples}`` placeholders. If no client is supplied, an OpenAI
    client is created from the installed ``openai`` package and environment
    variables such as ``OPENAI_API_KEY``.
    """

    _validate_generation_inputs(training_data, text_column, label_column, total, n_per_prompt)
    rng_seed = random_state
    positive_pool = training_data[training_data[label_column] == positive_label]
    negative_pool = training_data[training_data[label_column] == negative_label]
    if len(positive_pool) < positive_examples:
        raise ValueError("not enough positive examples in training_data")
    if len(negative_pool) < negative_examples:
        raise ValueError("not enough negative examples in training_data")

    llm_client = client or _openai_client()
    prompt_template = prompt or DEFAULT_PROMPT
    outputs: list[str] = []
    calls_needed = math.ceil(total / n_per_prompt)

    for call_idx in range(calls_needed):
        seed = None if rng_seed is None else rng_seed + call_idx
        pos = positive_pool.sample(positive_examples, replace=False, random_state=seed)
        neg = negative_pool.sample(negative_examples, replace=False, random_state=None if seed is None else seed + 10_000)
        rendered_prompt = prompt_template.format(
            n=min(n_per_prompt, total - len(outputs)),
            positive_examples=_format_examples(pos[text_column].astype(str).tolist()),
            negative_examples=_format_examples(neg[text_column].astype(str).tolist()),
        )

        response_text = _complete_with_retries(
            llm_client,
            model=model,
            prompt=rendered_prompt,
            temperature=temperature,
            max_attempts=max_attempts,
        )
        outputs.extend(_split_generated_texts(response_text, n_per_prompt))
        if len(outputs) >= total:
            break

    return outputs[:total]


def _validate_generation_inputs(
    training_data: pd.DataFrame,
    text_column: str,
    label_column: str,
    total: int,
    n_per_prompt: int,
) -> None:
    if text_column not in training_data.columns:
        raise ValueError(f"text_column '{text_column}' is not in training_data")
    if label_column not in training_data.columns:
        raise ValueError(f"label_column '{label_column}' is not in training_data")
    if total <= 0:
        raise ValueError("total must be positive")
    if n_per_prompt <= 0:
        raise ValueError("n_per_prompt must be positive")


def _openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install OpenAI support with: pip install 'sfrsa[llm]'") from exc
    return OpenAI()


def _format_examples(texts: Sequence[str]) -> str:
    return "\n".join(f"{idx}. {text}" for idx, text in enumerate(texts, start=1))


def _complete_with_retries(
    client: Any,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_attempts: int,
) -> str:
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - depends on provider errors
            last_error = exc
    raise RuntimeError("LLM generation failed after retries") from last_error


def _split_generated_texts(response_text: str, expected: int) -> list[str]:
    cleaned = response_text.replace("```", "").strip()
    if expected == 1:
        return [cleaned] if cleaned else []

    parts = [
        part.strip(" \n\t-")
        for part in re.split(r"(?:^|\n)\s*(?:\d+[\).\:]|\-)\s+", cleaned)
        if part.strip(" \n\t-")
    ]
    return parts if parts else [cleaned]
