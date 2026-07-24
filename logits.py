from typing import Dict, Tuple

import torch
from transformer_lens import HookedTransformer


def _compute_all_components_logits(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    モデルの全てのコンポーネントの logits を計算する関数.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 各層の logits とキャッシュ
    """
    # 各層の出力を logit へ変換
    logits = _compute_layer_logits_contribution(model, cache)

    # 各 Head の logits を計算
    head_logits = _compute_heads_logits_contribution(model, cache)

    return logits, head_logits


def _compute_layer_logits_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    各 layer の出力を logit へ変換する関数.

    Note: この関数は各層の出力に Final Layer Normalization を適用してから Unembedding を行う.
    これにより実際の GPT-2 モデルの処理フローと一致する正確な寄与度が計算される.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ

    Returns:
        torch.Tensor: [n_layers, d_vocab]
    """
    with torch.no_grad():
        # Unembedding 行列とバイアス項を取得
        W_U = model.W_U  # [d_model, d_vocab]
        b_U = getattr(model, "b_U", None)  # [d_vocab] (存在しない場合は None)
        n_layers = model.cfg.n_layers

        # 各層の出力を取得してスタック
        layer_outputs = [
            cache[f"blocks.{i}.hook_resid_post"][0, -1] for i in range(n_layers)
        ]
        layer_out = torch.stack(layer_outputs, dim=0)

        # Final Layer Normalizationを適用
        normalized_outputs = [
            model.ln_final(layer_out[i].unsqueeze(0)).squeeze(0)
            for i in range(n_layers)
        ]
        normalized_layer_out = torch.stack(normalized_outputs, dim=0)

        # einsum を使用して効率的に logits 計算: normalized_layer_out @ W_U
        # normalized_layer_out: [n_layers, d_model], W_U: [d_model, d_vocab]
        # 結果: [n_layers, d_vocab]
        logits = torch.einsum("lm,mv->lv", normalized_layer_out, W_U)

        # バイアス項が存在する場合は各層に追加
        if b_U is not None:
            logits = logits + b_U.unsqueeze(0)

        return logits


def _compute_heads_logits_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    モデル内の全ての Head の logits を計算する関数.

    計算の流れ:
    1. モデル内の全ての層の Head の Attention 出力 z を取得してスタック
    2. 全ての Head に出力重み W_O を一度に適用
    3. Unembedding 行列 W_U を適用して logits を計算

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ

    Returns:
        torch.Tensor: 各 Head の logits tensor [n_layers, n_heads, d_vocab]
    """
    with torch.no_grad():
        n_layers = model.cfg.n_layers

        # 全層の Attention 出力を取得
        z = torch.stack(
            [cache[f"blocks.{i}.attn.hook_z"][0, -1] for i in range(n_layers)], dim=0
        )

        # 出力重み行列 ([n_layers, n_heads, d_head, d_model]) を取得
        W_O = model.W_O

        # Unembedding 行列とバイアス項を取得
        W_U = model.W_U  # [d_model, d_vocab]
        b_U = getattr(model, "b_U", None)  # [d_vocab] (存在しない場合は None)

        # 各 Head の出力を残差ストリーム次元に変換: z @ W_O
        # z: [n_layers, n_heads, d_head], W_O: [n_layers, n_heads, d_head, d_model]
        # 結果: [n_layers, n_heads, d_model]
        head_outputs = torch.einsum("lhd,lhdm->lhm", z, W_O)

        # Final Layer Normalizationを適用
        n_layers, n_heads, _ = head_outputs.shape
        normalized_outputs = []
        for layer_idx in range(n_layers):
            layer_normalized = []
            for head_idx in range(n_heads):
                normalized = model.ln_final(
                    head_outputs[layer_idx, head_idx].unsqueeze(0)
                ).squeeze(0)
                layer_normalized.append(normalized)
            normalized_outputs.append(torch.stack(layer_normalized, dim=0))
        normalized_head_outputs = torch.stack(normalized_outputs, dim=0)

        # einsum を使用して効率的に logits 計算: normalized_head_outputs @ W_U
        # normalized_head_outputs: [n_layers, n_heads, d_model], W_U: [d_model, d_vocab]
        # 結果: [n_layers, n_heads, d_vocab]
        logits = torch.einsum("lhm,mv->lhv", normalized_head_outputs, W_U)

        # バイアス項が存在する場合は各層・各ヘッドに追加
        if b_U is not None:
            logits = logits + b_U.unsqueeze(0).unsqueeze(0)

        return logits
