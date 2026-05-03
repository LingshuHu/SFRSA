import numpy as np
import pandas as pd

from sfrsa import compute_utility_scores, select_augmented_texts, train_step0_utility_model


def test_step0_utility_scores_with_example_data():
    data = pd.read_csv("example_data_step0_training.csv")
    model = train_step0_utility_model(
        data,
        text_column="review2",
        label_column="label",
        validation_size=0.2,
        random_state=7,
        max_features=500,
    )

    candidates = data[data["label"] == 1].head(5)
    scores = compute_utility_scores(model, candidates, text_column="review2", candidate_label=1)

    assert scores.shape == (len(candidates),)

    selected = select_augmented_texts(
        candidates["review2"].tolist(),
        candidate_embeddings=np.eye(len(candidates)),
        method="iu-dpp",
        budget=2,
        utility_scores=scores,
    )
    assert len(selected) == 2


def test_step0_rejects_unknown_model_type():
    data = pd.read_csv("example_data_step0_training.csv")
    try:
        train_step0_utility_model(
            data,
            model_type="svm",
            text_column="review2",
            label_column="label",
        )
    except ValueError as exc:
        assert "model_type" in str(exc)
    else:
        raise AssertionError("expected ValueError")
