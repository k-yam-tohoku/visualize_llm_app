import csv
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformer_lens import HookedTransformer


def get_random_prompt(csv_path: str = "data/prompt_sample.csv") -> Tuple[str, str]:
    """
    CSV ファイルからランダムにプロンプトと対応する object を1つ取得する.

    Args:
        csv_path (str): CSV ファイルのパス.

    Returns:
        Tuple[str, str]: (プロンプト文字列, object 文字列) のタプル.
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


def get_prompt_samples(csv_path: str = "data/prompt_sample.csv") -> list[dict[str, str]]:
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


def check_answer_correctness(
    model: HookedTransformer,
    prompt: str,
    logits: torch.Tensor,
    expected_answer: str,
) -> Optional[bool]:
    """
    モデルの出力と期待される答えが一致するかを文脈を考慮してチェックする関数.

    プロンプトと期待される答えを結合してトークン化することで文脈に依存するトークン分割を考慮した正確な比較を行う.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        prompt (str): モデルに入力されたプロンプト.
        logits (torch.Tensor): モデルの出力ロジット.
        expected_answer (str): 期待される答えの文字列.

    Returns:
        Optional[bool]: 一致判定結果.
            - None の場合は期待される答えが空文字列.
            - True の場合は一致, False の場合は不一致.
    """
    if not expected_answer:
        return None

    # モデル出力の最初のトークン ID (logits から直接取得)
    predicted_first_token_id = logits[0, -1].argmax().item()

    # プロンプト + 期待される答えの形でトークン化 (文脈を考慮)
    # スペース有り/無しの両パターンで試行
    full_text_with_space = prompt + " " + expected_answer
    full_text_without_space = prompt + expected_answer

    # プロンプト部分のトークン数を取得
    prompt_tokens = model.to_tokens(prompt, prepend_bos=False)[0]
    prompt_length = len(prompt_tokens)

    # 文脈を考慮した期待される答えの最初のトークン ID を取得
    full_tokens_with_space = model.to_tokens(full_text_with_space, prepend_bos=False)[0]
    full_tokens_without_space = model.to_tokens(
        full_text_without_space, prepend_bos=False
    )[0]

    # プロンプトの後の最初のトークン ID を取得
    expected_first_token_id_with_space = None
    expected_first_token_id_without_space = None

    if len(full_tokens_with_space) > prompt_length:
        expected_first_token_id_with_space = full_tokens_with_space[
            prompt_length
        ].item()

    if len(full_tokens_without_space) > prompt_length:
        expected_first_token_id_without_space = full_tokens_without_space[
            prompt_length
        ].item()

    # どちらかのパターンと一致しているかチェック
    is_correct = (
        expected_first_token_id_with_space is not None
        and predicted_first_token_id == expected_first_token_id_with_space
    ) or (
        expected_first_token_id_without_space is not None
        and predicted_first_token_id == expected_first_token_id_without_space
    )

    return is_correct
