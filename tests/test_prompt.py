from types import SimpleNamespace

import torch

import backend
from prompt import check_answer_correctness, get_expected_token_ids


class FakeModel:
    cfg = SimpleNamespace(d_vocab=4, n_layers=1, n_heads=1)

    def to_tokens(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        del prepend_bos
        tokenizations = {
            "Hello": [0],
            "Hello ,": [0, 2],
            "Hello,": [0, 1],
        }
        return torch.tensor([tokenizations[text]])


def test_expected_token_ids_include_space_variants_without_duplicates():
    model = FakeModel()

    assert get_expected_token_ids(model, "Hello", ",") == [2, 1]


def test_correctness_and_ranks_use_the_same_token_candidates():
    model = FakeModel()
    output_logits = torch.tensor([[[0.0, 10.0, 2.0, 5.0]]])
    layer_logits = torch.tensor([[0.0, 10.0, 2.0, 5.0]])
    head_logits = torch.tensor([[[0.0, 3.0, 8.0, 5.0]]])

    assert check_answer_correctness(model, "Hello", output_logits, ",") is True

    ranks = backend.calculate_ranks(
        model,
        "Hello",
        ",",
        layer_logits,
        head_logits,
    )

    assert ranks["MLP0"] == 1
    assert ranks["A0.H0"] == 1
    assert ranks["Output"] == 1
