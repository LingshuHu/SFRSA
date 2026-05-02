import pandas as pd

from sfrsa.generation import generate_few_random_shot


class _FakeCompletions:
    def create(self, **kwargs):
        class Message:
            content = "1. generated positive text\n2. another generated text"

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


class _FakeClient:
    class chat:
        completions = _FakeCompletions()


def test_generate_few_random_shot_with_client():
    data = pd.DataFrame(
        {
            "text": ["p1", "p2", "p3", "n1", "n2", "n3"],
            "label": [1, 1, 1, 0, 0, 0],
        }
    )

    outputs = generate_few_random_shot(
        data,
        total=2,
        positive_examples=2,
        negative_examples=2,
        n_per_prompt=2,
        client=_FakeClient(),
        random_state=1,
    )

    assert outputs == ["generated positive text", "another generated text"]
