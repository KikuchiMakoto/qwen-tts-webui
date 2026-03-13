# Qwen3-TTS WebUI

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) を使用したカスタムボイス音声合成 WebUI です。
Streamlit ベースの日本語インターフェースで、ボイスクローニングと音声合成を簡単に行えます。

## 機能

### カスタムボイスモデル学習モード
- リファレンス音声（wav, mp3, flac, ogg, m4a）をアップロードして、カスタムボイスモデルを作成
- ニックネーム付きで最大10個までキャッシュ保存
- ボイスモデルのエクスポート／インポート（二次利用可能）
- 選択言語に対応した挨拶音声のプレビュー再生

### 音声合成モード
- 保存済みまたはアップロードしたボイスモデルを使用して音声合成
- チャット形式のインターフェースで対話的にテキストを入力・音声を生成
- 合成音声の再生およびダウンロード
- 生成中の進捗表示

### カスタム音声即時合成モード
- リファレンス音声から直接音声合成（ボイスモデルのキャッシュなし）
- 学習から合成までを一連で実行
- チャット形式のインターフェース

## 対応言語

日本語（デフォルト）、中国語、英語、韓国語、ドイツ語、フランス語、ロシア語、ポルトガル語、スペイン語、イタリア語

## セットアップ

### 必要要件

- Python 3.10〜3.12
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/KikuchiMakoto/qwen-tts-webui.git
cd qwen-tts-webui

# 依存関係をインストール（CPU版PyTorch）
uv sync
```

### GPU（CUDA）を使用する場合

デフォルトでは CPU で動作しますが、CUDA 対応 GPU がある場合は自動的に GPU を使用します。
CUDA 版 PyTorch を明示的にインストールするには:

```bash
# CUDA 12.4
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.8
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

GPU 使用時は FlashAttention 2 のインストールを推奨します（VRAM 使用量が削減されます）:

```bash
uv pip install flash-attn --no-build-isolation
```

### ROCm（AMD GPU）環境

ROCm 版 PyTorch をインストールすることで、AMD GPU でも音声生成が可能です:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

ROCm 版 PyTorch では `torch.cuda.is_available()` が `True` を返すため、コードの変更なしで自動的に GPU が使用されます。

### CPU を強制する場合

GPU がある環境でも CPU を使用したい場合は、環境変数 `FORCE_CPU=1` を設定してください:

```bash
FORCE_CPU=1 uv run streamlit run app.py
```

## 起動

```bash
uv run streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## 使用するモデル

| モデル | サイズ | 説明 |
|--------|--------|------|
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B（デフォルト） | 高品質な3秒ボイスクローニング |
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | 軽量版ボイスクローニング |

モデルは初回使用時に自動ダウンロードされます。

## プロジェクト構成

```
qwen-tts-webui/
├── app.py           # Streamlit UI（メインアプリケーション）
├── engine.py        # Qwen3-TTS エンジンラッパー
├── voice_store.py   # ボイスモデル永続化管理
├── pyproject.toml   # uv プロジェクト設定
└── README.md
```

## ライセンス

本プロジェクトは Qwen3-TTS モデルを使用しています。モデルの利用規約については [Qwen3-TTS リポジトリ](https://github.com/QwenLM/Qwen3-TTS) を参照してください。
