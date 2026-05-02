# SFRSA

Selective Few-Random-Shot Augmentation (SFRSA) is a two-stage text augmentation workflow:

1. Over-generate candidate minority-class texts with few-random-shot LLM prompting.
2. Select a fixed-size subset that balances minority relevance and diversity.

The package exposes two main functions:

- `select_augmented_texts`: select generated texts using `kmeans`, `dpp`, or `iu-dpp`.
- `generate_few_random_shot`: call an LLM API to generate texts from random positive and negative examples.

## Install

```bash
pip install sfrsa
```

For OpenAI generation support:

```bash
pip install "sfrsa[llm]"
```

## Select Generated Texts

```python
import pandas as pd
from sfrsa import select_augmented_texts

df = pd.read_csv("example_data.csv")
generated = df[df["type"] == "generated"]
minority = df[df["type"] == "original"]

selected = select_augmented_texts(
    candidates=generated,
    minority_texts=minority,
    method="dpp",
    budget=20,
    text_column="text",
    alpha=1.0,
    gamma=2.0,
    random_state=42,
)

print(selected.head())
```

If you already computed BERT or sentence-transformer embeddings, pass them directly:

```python
selected = select_augmented_texts(
    candidates=generated,
    minority_texts=minority,
    candidate_embeddings=candidate_embeddings,
    minority_embeddings=minority_embeddings,
    method="kmeans",
    budget=100,
)
```

For IU-DPP, provide training-aware utility scores for the candidates:

```python
selected = select_augmented_texts(
    candidates=generated,
    minority_texts=minority,
    candidate_embeddings=candidate_embeddings,
    method="iu-dpp",
    budget=100,
    utility_scores=utility_scores,
    beta=1.0,
    gamma=1.0,
)
```

## Generate Few-Random-Shot Candidates

Set `OPENAI_API_KEY` in your environment or pass a configured OpenAI client.

```python
import pandas as pd
from sfrsa import generate_few_random_shot

train = pd.read_csv("training.csv")

texts = generate_few_random_shot(
    training_data=train,
    text_column="text",
    label_column="label",
    positive_label=1,
    negative_label=0,
    total=50,
    positive_examples=10,
    negative_examples=10,
    n_per_prompt=1,
    model="gpt-4.1-mini",
    prompt=(
        "Write {n} movie review(s) within 170 words that are similar to the "
        "positive examples but different from the negative examples.\n\n"
        "Positive examples:\n{positive_examples}\n\n"
        "Negative examples:\n{negative_examples}"
    ),
)
```

## Publishing

This repository is ready for GitHub and PyPI after you choose the final project metadata in `pyproject.toml`.

```bash
python -m build
twine upload dist/*
```

GitHub publishing requires creating a repository under your account and pushing this working tree.
