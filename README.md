# SFRSA

Selective Few-Random-Shot Augmentation (SFRSA) is a two-stage text augmentation workflow:

1. Over-generate candidate minority-class texts with few-random-shot LLM prompting.
2. Select a fixed-size subset that balances minority relevance and diversity.

The package exposes three main workflows:

- `select_augmented_texts`: select generated texts using `kmeans`, `dpp`, or `iu-dpp`.
- `generate_few_random_shot`: call an LLM API to generate texts from random positive and negative examples.
- `train_step0_utility_model` and `compute_utility_scores`: train a Step-0 classifier and score candidates for IU-DPP.

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
    embedding_model_name="bert-base-uncased",
    alpha=1.0,
    gamma=2.0,
    random_state=42,
)

print(selected.head())
```

By default, selection computes BERT embeddings by summing the last four hidden layers and averaging over tokens. The most important selection parameters are:

- `method`: selection strategy. Use `"kmeans"` for centroid-guided cluster coverage, `"dpp"` for relevance plus diversity, or `"iu-dpp"` for utility-aware DPP.
- `budget`: number of generated texts to return.
- `text_column`: dataframe column containing text.
- `alpha`: for DPP, larger values favor texts closer to the minority-class centroid.
- `beta`: for IU-DPP, larger values favor texts with higher Step-0 utility scores.
- `gamma`: larger values increase diversity pressure in the DPP kernel.
- `candidate_embeddings` and `minority_embeddings`: optional precomputed embeddings. Use these when you want to reuse cached BERT outputs or embeddings from another encoder.
- `embedding_model_name`, `embedding_max_length`, `embedding_batch_size`, `embedding_device`: BERT embedding settings used when embeddings are not supplied.

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

## Step-0 Utility Scores

IU-DPP uses utility scores that estimate whether a candidate is useful for the downstream classifier. The package includes two Step-0 backends:

- `model_type="logistic"`: fast class-weighted logistic regression over TF-IDF features.
- `model_type="bert"`: fine-tuned BERT sequence classifier with classifier-head gradient utility. This uses CPU by default; set `device="cuda"` if you have a compatible GPU.

```python
import pandas as pd
from sfrsa import compute_utility_scores, train_step0_utility_model

train = pd.read_csv("example_data_step0_training.csv")

step0 = train_step0_utility_model(
    train,
    model_type="bert",
    text_column="review2",
    label_column="label",
    positive_label=1,
    negative_label=0,
    validation_size=0.1,
    bert_model_name="bert-base-uncased",
    bert_max_length=100,
    bert_batch_size=8,
    bert_epochs=3,
    device="cpu",
    random_state=42,
)

utility_scores = compute_utility_scores(
    step0,
    candidates=generated,
    text_column="text",
    candidate_label=1,
)
```

Important Step-0 parameters:

- `model_type`: `"logistic"` or `"bert"`.
- `positive_label` and `negative_label`: labels defining the target minority/positive class and contrast class.
- `validation_size`: held-out split used to compute the validation-loss gradient.
- `candidate_label`: label assigned to generated candidates when computing candidate gradients; usually the minority/positive label.
- `bert_model_name`: Hugging Face model name for the BERT backend.
- `bert_max_length`: maximum token length for BERT tokenization.
- `bert_batch_size`, `bert_epochs`, `bert_learning_rate`: BERT fine-tuning settings.
- `device`: `"cpu"` by default. Use `"cuda"` only when PyTorch can access a GPU.

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

Generation parameters:

- `total`: total number of generated texts to return.
- `positive_examples` and `negative_examples`: how many examples to sample into each prompt.
- `n_per_prompt`: how many texts the LLM should generate per API call.
- `prompt`: template with `{n}`, `{positive_examples}`, and `{negative_examples}` placeholders.
- `model` and `temperature`: LLM model name and sampling temperature.
- `random_state`: makes few-shot example sampling reproducible.
