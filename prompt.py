import csv
import random
from pathlib import Path

import torch
from transformer_lens import HookedTransformer


def get_random_prompt(csv_path: str = "data/prompt_sample.csv") -> tuple[str, str]:
    """
    CSV ファイルからランダムにプロンプトと対応する object を1つ取得する.

    Args:
        csv_path (str): CSV ファイルのパス.

    Returns:
        tuple[str, str]: (プロンプト文字列, object 文字列) のタプル.
    """
    csv_file = Path(csv_path)

    prompt_data = []
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompt_data.append(
                {"prompt": row["prompt"].strip(), "object": row["object"].strip()}
            )

    selected = random.choice(prompt_data)
    return selected["prompt"], selected["object"]


def get_prompt_samples(
    csv_path: str = "data/prompt_sample.csv",
) -> list[dict[str, str]]:
    """
    CSV ファイルからサンプルプロンプト一覧を取得する.

    Args:
        csv_path (str): CSV ファイルのパス.

    Returns:
        list[dict[str, str]]: サンプル一覧.
    """
    csv_file = Path(csv_path)
    samples = []
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            samples.append(
                {
                    "prompt": row["prompt"].strip(),
                    "subject": row["subject"].strip(),
                    "expected_answer": row["object"].strip(),
                    "keywords": row["keywords"].strip(),
                }
            )
    return samples


def get_expected_token_ids(
    model: HookedTransformer,
    prompt: str,
    expected_answer: str,
) -> list[int]:
    """Return unique first-token candidates with and without a separating space."""
    if not expected_answer:
        return []

    prompt_length = len(model.to_tokens(prompt, prepend_bos=False)[0])
    token_ids = []
    for separator in (" ", ""):
        full_tokens = model.to_tokens(
            prompt + separator + expected_answer,
            prepend_bos=False,
        )[0]
        if len(full_tokens) > prompt_length:
            token_ids.append(int(full_tokens[prompt_length].item()))

    return list(dict.fromkeys(token_ids))


def check_answer_correctness(
    model: HookedTransformer,
    prompt: str,
    logits: torch.Tensor,
    expected_answer: str,
) -> bool | None:
    """
    モデルの出力と期待される答えが一致するかを文脈を考慮してチェックする関数.

    プロンプトと期待される答えを結合してトークン化することで文脈に依存するトークン分割を考慮した正確な比較を行う.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        prompt (str): モデルに入力されたプロンプト.
        logits (torch.Tensor): モデルの出力ロジット.
        expected_answer (str): 期待される答えの文字列.

    Returns:
        bool | None: 一致判定結果.
            - None の場合は期待される答えが空文字列.
            - True の場合は一致, False の場合は不一致.
    """
    if not expected_answer:
        return None

    predicted_first_token_id = logits[0, -1].argmax().item()
    expected_token_ids = get_expected_token_ids(model, prompt, expected_answer)
    return predicted_first_token_id in expected_token_ids
