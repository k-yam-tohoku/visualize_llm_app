import torch
from transformer_lens import ActivationCache, HookedTransformer


def get_cache(
    model: HookedTransformer,
    prompt: str,
) -> tuple[torch.Tensor, dict | ActivationCache]:
    """
    モデルのキャッシュを取得する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        prompt (str): モデルに入力するプロンプト.

    Returns:
        tuple[torch.Tensor, dict | ActivationCache]: モデルの出力とキャッシュ.
    """
    token_ids = model.to_tokens(prompt, prepend_bos=False).cpu()
    with torch.no_grad():
        logits, cache = model.run_with_cache(token_ids)
    cache.model = None
    return logits, cache


def get_output(model: HookedTransformer, logits: torch.Tensor) -> str:
    """
    モデルの出力を取得する関数.
    ここでは, logits の最後のトークン位置の top1 トークンを取得して文字列に変換する.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        logits (torch.Tensor): モデルの出力ロジット.

    Returns:
        str: 予測された次のトークン文字列.
    """
    top1_token_id = logits[0, -1].argmax().item()
    return model.to_string(top1_token_id)
