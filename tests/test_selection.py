import numpy as np
import pandas as pd

from sfrsa import select_augmented_texts


def test_select_dpp_from_dataframe():
    candidates = pd.DataFrame(
        {
            "text": ["good film", "great movie", "bad film", "excellent story"],
            "type": ["generated"] * 4,
        }
    )
    minority = pd.DataFrame({"text": ["good story", "great film"]})
    cand_emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.8, 0.2]])
    min_emb = np.array([[1.0, 0.0], [0.9, 0.1]])

    selected = select_augmented_texts(
        candidates,
        minority,
        candidate_embeddings=cand_emb,
        minority_embeddings=min_emb,
        method="dpp",
        budget=2,
    )

    assert isinstance(selected, pd.DataFrame)
    assert len(selected) == 2
    assert set(selected["text"]).issubset(set(candidates["text"]))


def test_select_kmeans_with_embeddings():
    candidates = ["a", "b", "c", "d"]
    cand_emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    min_emb = np.array([[1.0, 0.0]])

    selected = select_augmented_texts(
        candidates,
        candidate_embeddings=cand_emb,
        minority_embeddings=min_emb,
        method="kmeans",
        budget=2,
        random_state=0,
    )

    assert len(selected) == 2


def test_iu_dpp_requires_utilities():
    try:
        select_augmented_texts(["a", "b"], method="iu-dpp", budget=1)
    except ValueError as exc:
        assert "utility_scores" in str(exc)
    else:
        raise AssertionError("expected ValueError")
