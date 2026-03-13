"""ボイスモデル永続化 - 保存・読込・エクスポート・インポート."""

import json
from datetime import datetime
from pathlib import Path

import torch

STORE_DIR = Path("voice_models")
METADATA_FILE = STORE_DIR / "metadata.json"
MAX_CACHED = 10


def _ensure_store():
    """ストアディレクトリとメタデータファイルを初期化する。"""
    STORE_DIR.mkdir(exist_ok=True)
    if not METADATA_FILE.exists():
        METADATA_FILE.write_text("{}")


def _load_metadata() -> dict:
    _ensure_store()
    return json.loads(METADATA_FILE.read_text())


def _save_metadata(metadata: dict):
    _ensure_store()
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


def save_voice(nickname: str, prompt_items, language: str, model_size: str):
    """ボイスクローンプロンプトをニックネーム付きで保存する。

    最大 MAX_CACHED 個まで保存。超過時は最も古いモデルを自動削除する。
    """
    _ensure_store()
    metadata = _load_metadata()

    # キャッシュ上限の管理
    if len(metadata) >= MAX_CACHED and nickname not in metadata:
        oldest = min(metadata, key=lambda k: metadata[k]["created_at"])
        remove_voice(oldest)
        metadata = _load_metadata()

    filepath = STORE_DIR / f"{nickname}.pt"
    torch.save(prompt_items, filepath)

    metadata[nickname] = {
        "file": str(filepath),
        "language": language,
        "model_size": model_size,
        "created_at": datetime.now().isoformat(),
    }
    _save_metadata(metadata)


def load_voice(nickname: str):
    """保存済みボイスクローンプロンプトを読み込む。"""
    metadata = _load_metadata()
    if nickname not in metadata:
        raise ValueError(f"ボイスモデル '{nickname}' が見つかりません")
    filepath = Path(metadata[nickname]["file"])
    return torch.load(filepath, weights_only=False)


def list_voices() -> list[dict]:
    """保存済みボイスモデルの一覧を返す（新しい順）。"""
    metadata = _load_metadata()
    voices = []
    for name, info in sorted(
        metadata.items(), key=lambda x: x[1]["created_at"], reverse=True
    ):
        voices.append({"nickname": name, **info})
    return voices


def remove_voice(nickname: str):
    """保存済みボイスモデルを削除する。"""
    metadata = _load_metadata()
    if nickname in metadata:
        filepath = Path(metadata[nickname]["file"])
        if filepath.exists():
            filepath.unlink()
        del metadata[nickname]
        _save_metadata(metadata)


def export_voice(nickname: str) -> bytes:
    """ボイスモデルをバイト列としてエクスポートする（ダウンロード用）。"""
    metadata = _load_metadata()
    if nickname not in metadata:
        raise ValueError(f"ボイスモデル '{nickname}' が見つかりません")
    filepath = Path(metadata[nickname]["file"])
    return filepath.read_bytes()


def import_voice(nickname: str, data: bytes, language: str, model_size: str):
    """バイト列からボイスモデルをインポートする。"""
    _ensure_store()
    filepath = STORE_DIR / f"{nickname}.pt"
    filepath.write_bytes(data)

    metadata = _load_metadata()
    if len(metadata) >= MAX_CACHED and nickname not in metadata:
        oldest = min(metadata, key=lambda k: metadata[k]["created_at"])
        remove_voice(oldest)
        metadata = _load_metadata()

    metadata[nickname] = {
        "file": str(filepath),
        "language": language,
        "model_size": model_size,
        "created_at": datetime.now().isoformat(),
    }
    _save_metadata(metadata)
