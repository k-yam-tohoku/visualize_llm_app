# Visualize LLM App

このアプリケーションは，大規模言語モデル (LLM) の内部動作を可視化する Web アプリです．<br>
特定の入力に対して，モデルがどのように予測を行っているかを，**Attention Pattern** や **中間層の出力** といった観点から視覚的に理解できます．

本アプリは，東北大学オープンキャンパスにおける研究室展示の一部として開発されたデモ用アプリケーションです．

<p align="center">
  <img src="figures/samples/model_sample.png" width="70%" alt="モデル構造の可視化">
</p>

<p align="center">
  <img src="figures/samples/attention_map_sample.png" width="35%" alt="Attention パターンの可視化">
  <img src="figures/samples/logits_sample.png" width="35%" alt="Logits の可視化">
</p>

## 🔍 主な機能

- 任意のプロンプト入力に対する LLM の出力可視化
- Attention Pattern の可視化（各層・各ヘッド）
- MLP および Attention 出力の表示

## 🛠️ セットアップ方法（uv 推奨）

### 1. リポジトリのクローン

```bash
git clone https://github.com/Fukata-K/visualize_llm_app.git
cd visualize_llm_app
```

### 2. Python 環境の構築（uv）

[uv](https://docs.astral.sh/uv/) で依存関係をインストールします（Python 3.12）．

```bash
uv sync
```

### 3. React + FastAPI 版の起動（開発）

バックエンドを起動します．

```bash
uv run uvicorn backend:app --reload --host 127.0.0.1 --port 8000
```

別のターミナルでフロントエンドを起動します．

```bash
cd frontend
npm install
npm run dev
```

ブラウザで `http://127.0.0.1:5173` を開きます．

React 版では，解析ジョブの進捗をポーリングし，生成済みの Attention Head / MLP / Output ノードから順にクリックできるようになります．全 transformer layer の処理完了を待つ必要はありません．各ノードの詳細（Attention Pattern / 予測ランキング）は画像ではなく JSON で返され，ブラウザ側で描画されます．

### 4. 本番配信（uvicorn のみ）

フロントエンドをビルドしておくと，FastAPI がその成果物（`frontend/dist`）を同一オリジンで配信します．uvicorn 1 プロセスだけで UI と API の両方を提供できます．

```bash
cd frontend
npm run build          # frontend/dist を生成
cd ..
uv run uvicorn backend:app --host 0.0.0.0 --port 8000
```

ブラウザで `http://localhost:8000` を開きます．

- ビルド時は `frontend/.env.production` により，API 呼び出しが相対パス（同一オリジン）になります（CORS 不要）．
- 解析ジョブはメモリ上，モデルはプロセスグローバルに保持されるため，**ワーカーは 1 のまま**（`--workers` を付けない）で運用してください．

### 5. アプリの終了

各ターミナルで `Ctrl + C` を押して終了してください（ブラウザを閉じるだけでは終了されません）．


## ▶️ アプリの使い方

### 1. プロンプト入力

任意のプロンプトを入力します．<br>
予測させたい部分が次の単語になるような文にしてください．推奨言語は英語です．<br>
`Ex. Sendai is located in the country of`

💡 **迷ったら `Random` を押してください．**<br>
`Random` でランダムなサンプル，`Samples` で一覧からプロンプトと答えを選んで自動入力できます．

### 2. 答え入力

入力したプロンプトに対応する正しい答えを入力します．<br>
`Ex. Japan`

### 3. 実行
`Go` を押すと言語モデルの内部動作を表すグラフが表示されます．

## 📁 ファイル構成（抜粋）

```
.
├── backend.py               # FastAPI バックエンド（API ＋ ビルド済みフロントの配信）
├── frontend/                # React + Vite フロントエンド
│ ├── src/                   #   main.tsx（UI 本体）, styles.css
│ ├── index.html
│ ├── vite.config.ts
│ ├── .env.production        #   本番ビルド用の API ベース（相対パス）
│ └── package.json
├── model.py                 # モデルのロードとキャッシュ
├── logits.py                # 各コンポーネントの logit 計算
├── prompt.py                # プロンプト処理とランダム選択
├── data/
│ ├── prompt_sample.csv      # ランダム選択用のプロンプトと回答
│ └── README.md              # データセットの説明
├── docs/                    # デモ用ガイド（PDF）
├── figures/samples/         # README 用のサンプル画像
├── pyproject.toml           # プロジェクト定義・依存関係（uv）
├── uv.lock                  # 依存関係のロックファイル
└── README.md                # 本ファイル
```

## ⚙️ 主なライブラリ

- Python 3.12（`requires-python >= 3.10`）
- 環境管理: uv
- バックエンド: FastAPI / Uvicorn / TransformerLens (v2.16.1) / PyTorch (v2.7.1)
- フロントエンド: React 18 / Vite 6 / TypeScript
