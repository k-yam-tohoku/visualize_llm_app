from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import get_act_name


def generate_attention_heatmaps(
    model: HookedTransformer,
    cache: dict | ActivationCache,
    prompt: str,
    output_dir: str = "figures/attention_patterns",
) -> None:
    """
    データセットの全レイヤー・全ヘッドについて平均 Attention Pattern の Heatmap を生成・保存する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        cache (dict | ActivationCache): attention cache.
        prompt (str): モデルに入力するプロンプト.
        output_dir (str): 出力ディレクトリのベースパス (default: "figures/attention_patterns").

    Returns:
        None
    """
    # モデル構成の取得
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads

    # プロンプトのトークン化
    tokens = model.to_str_tokens(prompt, prepend_bos=False)
    tokens = [token.replace(" ", "_") for token in tokens]
    seq_len = len(tokens)
    tick_positions = list(range(seq_len))

    # 出力ディレクトリの作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # レイヤーごとにバッチ処理
    for layer in range(num_layers):
        print(f"Processing Layer {layer}/{num_layers - 1}...")

        # レイヤーの Attention データを一度に取得して numpy 配列に変換
        layer_key = get_act_name("attn", layer)
        layer_attn_tensor = cache[layer_key][0]  # [num_heads, seq_len, seq_len]
        layer_attn = layer_attn_tensor.detach().cpu().numpy()

        # そのレイヤーの全ヘッドを処理
        for head in range(num_heads):
            save_path = f"{output_dir}/L{layer:02d}_H{head:02d}.png"
            attn_array = layer_attn[head]  # すでにnumpy配列

            # ヒートマップ作成
            _create_heatmap(attn_array, layer, head, tokens, tick_positions, save_path)


def _create_heatmap(
    attn_array: np.ndarray,
    layer: int,
    head: int,
    tokens: list,
    tick_positions: list,
    save_path: str,
) -> None:
    """
    ヒートマップ作成関数

    Args:
        attn_array (np.ndarray): attention weight 配列.
        layer (int): レイヤー番号.
        head (int): ヘッド番号.
        tokens (list): トークンリスト.
        tick_positions (list): 軸の目盛り位置.
        save_path (str): 保存パス.

    Returns:
        None
    """
    # matplotlib 設定の最適化
    fig, ax = plt.subplots(figsize=(6, 5))

    # イメージ表示
    im = ax.imshow(attn_array, cmap="Blues", aspect="equal", vmin=0, vmax=1)

    # カラーバーの追加
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight", fontsize=10)

    # 軸ラベルの設定
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key (Attended To)", fontsize=10)
    ax.set_ylabel("Query (Attending From)", fontsize=10)

    # タイトル
    ax.set_title(f"Layer {layer}, Head {head}", fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)  # 明示的な figure close
