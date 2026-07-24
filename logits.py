from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from transformer_lens import HookedTransformer


def _setup_matplotlib():
    """matplotlib の基本設定を行う関数."""
    import warnings

    # matplotlib 関連の警告を抑制
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["text.usetex"] = False

    # 日本語フォントの設定を試行
    try:
        import matplotlib.font_manager as fm

        japanese_fonts = ["Hiragino Sans", "Yu Gothic", "DejaVu Sans"]
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        for font in japanese_fonts:
            if font in available_fonts:
                plt.rcParams["font.family"] = [font]
                break
    except Exception:
        pass  # フォント設定に失敗しても続行


def _clean_token_text(text: str) -> str:
    """トークンテキストを表示用に整理し, 表示できない文字を除去する関数."""
    import re

    # 置換文字や制御文字を除去
    text = text.replace("\ufffd", "[?]")  # 置換文字を[?]に変換
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)  # 制御文字除去

    # RTL 言語 (Hebrew, Arabic 等) の文字を除去/置換
    text = re.sub(r"[\u0590-\u05FF]", "[HE]", text)  # Hebrew
    text = re.sub(r"[\u0600-\u06FF]", "[AR]", text)  # Arabic
    text = re.sub(r"[\u0700-\u074F]", "[SY]", text)  # Syriac
    text = re.sub(r"[\uFB1D-\uFDFF]", "[RTL]", text)  # Other RTL scripts
    text = re.sub(r"[\uFE70-\uFEFF]", "[RTL]", text)  # Arabic Presentation Forms

    # 特殊文字の簡単な置換
    text = text.replace("$", "USD").replace("_", "-")

    # 空白文字の処理
    if not text.strip():
        return "[SPACE]"

    # 長いテキストの短縮
    if len(text) > 15:
        return text[:15] + "..."

    return text


def calculate_all_logits_object_rank(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    prompt: str,
    object: str,
) -> Dict[str, int]:
    """
    モデルの全ての構成要素の object の出力順位を計算する関数.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル.
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ.
        prompt (str): モデルに入力されたプロンプト.
        object (str): 出力順位を計算する対象のトークン.

    Returns:
        Dict[str, int]: ノード名をキー, object の順位を値とする辞書.
    """
    layer_logits, head_logits = _compute_all_components_logits(model, cache)

    # オブジェクトのトークン ID を取得 (文脈を考慮した2パターン)
    full_text_with_space = prompt + " " + object
    full_text_without_space = prompt + object

    prompt_tokens = model.to_tokens(prompt, prepend_bos=False)[0]
    prompt_length = len(prompt_tokens)

    # スペース有り/無しでのトークン化
    full_tokens_with_space = model.to_tokens(full_text_with_space, prepend_bos=False)[0]
    full_tokens_without_space = model.to_tokens(
        full_text_without_space, prepend_bos=False
    )[0]

    # オブジェクトの最初のトークン ID を取得
    object_token_id = None
    if len(full_tokens_with_space) > prompt_length:
        object_token_id = full_tokens_with_space[prompt_length].item()
    elif len(full_tokens_without_space) > prompt_length:
        object_token_id = full_tokens_without_space[prompt_length].item()

    if object_token_id is None:
        # オブジェクトが見つからない場合は空の辞書を返す
        return {}

    ranks = {}

    # 各層の logits からオブジェクトの順位を計算
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    ranks["Input"] = model.cfg.d_vocab  # Input ノードの順位は vocab サイズ
    for layer_idx in range(n_layers):
        layer_logit = layer_logits[layer_idx]
        # 降順でソートしてオブジェクトの順位を取得
        sorted_indices = torch.argsort(layer_logit, descending=True)
        rank = (sorted_indices == object_token_id).nonzero(as_tuple=True)[0].item() + 1
        ranks[f"MLP{layer_idx}"] = rank

        # 各ヘッドの logits からオブジェクトの順位を計算
        for head_idx in range(n_heads):
            head_logit = head_logits[layer_idx, head_idx]
            sorted_indices = torch.argsort(head_logit, descending=True)
            rank = (sorted_indices == object_token_id).nonzero(as_tuple=True)[0]
            rank = rank.item() + 1
            ranks[f"A{layer_idx}.H{head_idx}"] = rank
    ranks["Output"] = ranks[f"MLP{n_layers - 1}"]  # Output ノードの順位は最終層の出力

    return ranks


def save_all_logits_figures(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
) -> None:
    """
    モデルの全ての層の logits を可視化し, 画像として保存する関数.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル.
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ.

    Returns:
        None
    """
    layer_logits, head_logits = _compute_all_components_logits(model, cache)

    for layer_idx in range(model.cfg.n_layers):
        print(f"Processing Layer {layer_idx}/{model.cfg.n_layers - 1}...")
        for head_idx in range(model.cfg.n_heads):
            _visualize_top_k_tokens(
                head_logits[layer_idx, head_idx],
                model,
                top_k=10,
                title=f"Layer {layer_idx} Head {head_idx} Logits",
                save_path=f"figures/logits/L{layer_idx:02d}_H{head_idx:02d}.png",
            )
        _visualize_top_k_tokens(
            layer_logits[layer_idx],
            model,
            top_k=10,
            title=f"Layer {layer_idx} Logits",
            save_path=f"figures/logits/L{layer_idx:02d}.png",
        )


def _visualize_top_k_tokens(
    logits: torch.Tensor,
    model: HookedTransformer,
    top_k: int = 10,
    title: str = "Top-K Token Logits",
    figsize: Tuple[int, int] = (4, 6),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
    font_size: int = 12,
) -> None:
    """
    単一の logits から上位 K 個のトークンを 1次元 Heatmap で可視化する関数.

    Args:
        logits (torch.Tensor): 可視化する logits [d_vocab]
        model (HookedTransformer): トークナイザーを含むモデル
        top_k (int): 表示する上位トークン数. Defaults to 10.
        title (str): グラフのタイトル. Defaults to "Top-K Token Logits".
        figsize (Tuple[int, int]): 図のサイズ (width, height). Defaults to (4, 6).
        cmap (str): カラーマップ名. Defaults to "Blues".
        save_path (Optional[str]): 保存パス. Defaults to None.
        font_size (int): フォントサイズ. Defaults to 10.

    Returns:
        None
    """
    _setup_matplotlib()

    # logits を numpy 配列に変換して上位 K 個を取得
    logits_np = (
        logits.detach().cpu().numpy()
        if isinstance(logits, torch.Tensor)
        else np.array(logits)
    )
    top_k_indices = np.argsort(logits_np)[-top_k:][::-1]
    top_k_values = logits_np[top_k_indices]

    # トークンテキストを取得・整理
    token_texts = []
    for idx in top_k_indices:
        try:
            token_text = model.tokenizer.decode([idx])
            token_text = _clean_token_text(token_text)
            token_texts.append(token_text)
        except Exception:
            token_texts.append(f"[UNK_{idx}]")

    # 図の作成
    fig, ax = plt.subplots(figsize=figsize)
    heatmap_data = top_k_values.reshape(-1, 1)

    vmax, vmin = heatmap_data.max(), heatmap_data.min()
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # カラーバーと軸設定
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Logit Value", fontsize=font_size - 2)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_yticks([])
    ax.set_xticks([])

    # テキスト表示
    for i, (value, token_text) in enumerate(zip(top_k_values, token_texts)):
        normalized_value = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.5
        text_color = "white" if normalized_value > 0.7 else "black"
        ax.text(
            0,
            i,
            token_text,
            ha="center",
            va="center",
            color=text_color,
            fontsize=font_size,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=font_size, pad=20)
    plt.tight_layout()

    # 保存
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


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
