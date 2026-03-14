"""ボイスモデル永続化 - 保存・読込・エクスポート・インポート."""

import io
import json
from datetime import datetime
from pathlib import Path

import torch

STORE_DIR = Path("voice_models")
MAX_CACHED = 10


def _get_model_dir(model_size: str) -> Path:
    """model_sizeに対応するサブディレクトリを返す。"""
    return STORE_DIR / model_size


def _get_metadata_file(model_size: str) -> Path:
    """model_sizeに対応するメタデータファイルを返す。"""
    return _get_model_dir(model_size) / "metadata.json"


def _ensure_store(model_size: str):
    """ストアディレクトリとメタデータファイルを初期化する。"""
    model_dir = _get_model_dir(model_size)
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = _get_metadata_file(model_size)
    if not metadata_file.exists():
        metadata_file.write_text("{}", encoding="utf-8")


def _load_metadata(model_size: str) -> dict:
    _ensure_store(model_size)
    metadata_file = _get_metadata_file(model_size)
    return json.loads(metadata_file.read_text(encoding="utf-8"))


def _save_metadata(model_size: str, metadata: dict):
    _ensure_store(model_size)
    metadata_file = _get_metadata_file(model_size)
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _get_model_size_from_data(data: bytes) -> str | None:
    """ptファイルのバイナリデータからmodel_sizeを推定する。"""
    try:
        loaded = torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")
        if isinstance(loaded, dict) and "model_size" in loaded:
            return loaded["model_size"]
        return None
    except Exception:
        return None


def save_voice(nickname: str, prompt_items, language: str, model_size: str):
    """ボイスクローンプロンプトをニックネーム付きで保存する。

    model_sizeごとにMAX_CACHED個まで保存。超過時は最も古いモデルを自動削除する。
    """
    _ensure_store(model_size)
    metadata = _load_metadata(model_size)

    # model_sizeごとのキャッシュ上限管理
    if nickname not in metadata:
        if len(metadata) >= MAX_CACHED:
            if metadata:
                oldest = min(metadata, key=lambda k: metadata[k]["created_at"])
                remove_voice(oldest, model_size)
                metadata = _load_metadata(model_size)

    filepath = _get_model_dir(model_size) / f"{nickname}.pt"
    torch.save(prompt_items, filepath)

    metadata[nickname] = {
        "file": str(filepath),
        "language": language,
        "model_size": model_size,
        "created_at": datetime.now().isoformat(),
    }
    _save_metadata(model_size, metadata)


def load_voice(nickname: str, model_size: str):
    """保存済みボイスクローンプロンプトを読み込む。"""
    metadata = _load_metadata(model_size)
    if nickname not in metadata:
        raise ValueError(f"ボイスモデル '{nickname}' が見つかりません（{model_size}）")
    filepath = Path(metadata[nickname]["file"])
    return torch.load(filepath, weights_only=False)


def list_voices() -> list[dict]:
    """すべての保存済みボイスモデルの一覧を返す（新しい順）。"""
    voices = []
    for model_size in ["1.7B", "0.6B"]:
        metadata = _load_metadata(model_size)
        for name, info in sorted(
            metadata.items(), key=lambda x: x[1]["created_at"], reverse=True
        ):
            voices.append({"nickname": name, **info})
    return voices


def list_voices_by_size(model_size: str) -> list[dict]:
    """指定されたmodel_sizeの保存済みボイスモデル一覧を返す（新しい順）。"""
    metadata = _load_metadata(model_size)
    voices = []
    for name, info in sorted(
        metadata.items(), key=lambda x: x[1]["created_at"], reverse=True
    ):
        voices.append({"nickname": name, **info})
    return voices


def remove_voice(nickname: str, model_size: str):
    """保存済みボイスモデルを削除する。"""
    metadata = _load_metadata(model_size)
    if nickname in metadata:
        filepath = Path(metadata[nickname]["file"])
        if filepath.exists():
            filepath.unlink()
        del metadata[nickname]
        _save_metadata(model_size, metadata)


def export_voice(nickname: str, model_size: str) -> tuple[bytes, str]:
    """ボイスモデルをバイト列としてエクスポートする（ダウンロード用）。

    Returns:
        (data, filename) タプル
    """
    metadata = _load_metadata(model_size)
    if nickname not in metadata:
        raise ValueError(f"ボイスモデル '{nickname}' が見つかりません（{model_size}）")
    filepath = Path(metadata[nickname]["file"])
    filename = f"{nickname}_{model_size}.pt"
    return filepath.read_bytes(), filename


def import_voice(nickname: str, data: bytes, language: str, model_size: str):
    """バイト列からボイスモデルをインポートする。

    Raises:
        ValueError: model_sizeがデータと一致しない場合
    """
    _ensure_store(model_size)

    # サイズ検証: 既存のデータとmodel_sizeが一致するか確認
    saved_model_size = _get_model_size_from_data(data)
    if saved_model_size and saved_model_size != model_size:
        raise ValueError(
            f"モデルサイズが一致しません。"
            f"データ: {saved_model_size}, 指定: {model_size}"
        )

    filepath = _get_model_dir(model_size) / f"{nickname}.pt"
    filepath.write_bytes(data)

    metadata = _load_metadata(model_size)
    if nickname not in metadata:
        if len(metadata) >= MAX_CACHED:
            if metadata:
                oldest = min(metadata, key=lambda k: metadata[k]["created_at"])
                remove_voice(oldest, model_size)
                metadata = _load_metadata(model_size)

    metadata[nickname] = {
        "file": str(filepath),
        "language": language,
        "model_size": model_size,
        "created_at": datetime.now().isoformat(),
    }
    _save_metadata(model_size, metadata)
