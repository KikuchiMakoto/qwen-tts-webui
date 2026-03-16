"""Qwen3-TTS Web API サーバー - VOICEVOX互換HTTP APIサーバー.

起動方法:
    uv run python api_server.py
    # オプション
    uv run python api_server.py --host 0.0.0.0 --port 50021 --model-size 1.7B
"""

import argparse
import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from engine import SUPPORTED_LANGUAGES, TTSEngine
from voice_store import (
    export_voice,
    list_voices_by_size,
    load_voice,
    remove_voice,
    save_voice,
)

# --- アプリケーション ---

app = FastAPI(
    title="Qwen3-TTS Web API",
    description="Qwen3-TTSを使用したVOICEVOX互換音声合成APIサーバー",
    version="0.1.0",
)

# グローバルエンジンインスタンス
_engine: TTSEngine | None = None
_default_model_size: str = "1.7B"

# デフォルトサンプリングレート（Qwen3-TTSの出力に合わせた値）
DEFAULT_SAMPLE_RATE: int = 24000

# multi_synthesis時のクエリ間の無音時間（秒）
INTER_QUERY_SILENCE_DURATION: float = 0.3


def get_engine() -> TTSEngine:
    """TTSEngineのシングルトンを返す。"""
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine


def _wav_to_bytes(wav: np.ndarray, sr: int) -> bytes:
    """numpy音声をWAVバイト列に変換する。"""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _speaker_id_to_nickname(speaker_id: int, model_size: str) -> str:
    """speaker_idをnicknameに変換する。

    speaker_idはlist_voices_by_sizeの返り値の0ベースインデックス。
    """
    voices = list_voices_by_size(model_size)
    if speaker_id < 0 or speaker_id >= len(voices):
        raise HTTPException(
            status_code=404,
            detail=f"話者ID {speaker_id} が見つかりません。",
        )
    return voices[speaker_id]["nickname"]


def _build_speaker_list(model_size: str) -> list[dict]:
    """保存済みボイスモデルをVOICEVOX形式のspeaker一覧に変換する。"""
    voices = list_voices_by_size(model_size)
    speakers = []
    for idx, voice in enumerate(voices):
        speakers.append(
            {
                "name": voice["nickname"],
                "speaker_uuid": f"qwen-tts-{voice['nickname']}",
                "styles": [
                    {
                        "name": "ノーマル",
                        "id": idx,
                        "type": "talk",
                    }
                ],
                "version": "0.1.0",
                "supported_features": {
                    "permitted_synthesis_morphing": "NOTHING",
                },
            }
        )
    return speakers


# --- Pydantic モデル ---


class Mora(BaseModel):
    """音素情報（VOICEVOX互換）."""

    text: str
    consonant: str | None = None
    consonant_length: float | None = None
    vowel: str
    vowel_length: float
    pitch: float


class AccentPhrase(BaseModel):
    """アクセント句（VOICEVOX互換）."""

    moras: list[Mora]
    accent: int
    pause_mora: Mora | None = None
    is_interrogative: bool = False


class AudioQuery(BaseModel):
    """音声合成クエリ（VOICEVOX互換 + Qwen3-TTS拡張）.

    VOICEVOX互換フィールド:
        accent_phrases: アクセント句リスト（本実装では未使用）
        speed_scale: 速度スケール（参考値）
        pitch_scale: ピッチスケール（参考値）
        intonation_scale: イントネーションスケール（参考値）
        volume_scale: 音量スケール（参考値）
        pre_phoneme_length: 前余白秒数（参考値）
        post_phoneme_length: 後余白秒数（参考値）
        output_sampling_rate: 出力サンプリングレート（参考値）
        output_stereo: ステレオ出力（参考値）

    Qwen3-TTS拡張フィールド:
        text: 合成テキスト
        language: 出力言語
        model_size: モデルサイズ
        temperature: 感情値
        repetition_penalty: 繰り返し抑制係数
        top_p: 核サンプリング確率閾値
        top_k: サンプリング候補数
    """

    accent_phrases: list[AccentPhrase] = Field(default_factory=list)
    speed_scale: float = 1.0
    pitch_scale: float = 0.0
    intonation_scale: float = 1.0
    volume_scale: float = 1.0
    pre_phoneme_length: float = 0.1
    post_phoneme_length: float = 0.1
    output_sampling_rate: int = DEFAULT_SAMPLE_RATE
    output_stereo: bool = False
    # Qwen3-TTS 拡張フィールド
    text: str = ""
    language: str = "Japanese"
    model_size: str = "1.7B"
    temperature: float = 0.65
    repetition_penalty: float = 1.15
    top_p: float = 0.9
    top_k: int = 50


class SpeakerStyle(BaseModel):
    """話者スタイル."""

    name: str
    id: int
    type: str = "talk"


class Speaker(BaseModel):
    """話者情報（VOICEVOX互換）."""

    name: str
    speaker_uuid: str
    styles: list[SpeakerStyle]
    version: str = "0.1.0"
    supported_features: dict = Field(
        default_factory=lambda: {"permitted_synthesis_morphing": "NOTHING"}
    )


class SpeakerInfo(BaseModel):
    """話者詳細情報（VOICEVOX互換）."""

    policy: str = ""
    portrait: str = ""
    style_infos: list[dict] = Field(default_factory=list)


class VoiceCreateResponse(BaseModel):
    """ボイス作成レスポンス."""

    speaker_id: int
    nickname: str
    language: str
    model_size: str


# --- API エンドポイント ---


@app.get("/version", response_model=str, tags=["System"])
async def get_version() -> str:
    """APIバージョンを返す。"""
    return "0.1.0"


@app.get("/supported_devices", tags=["System"])
async def get_supported_devices() -> dict:
    """サポートされているデバイス情報を返す。"""
    engine = get_engine()
    device_info = engine.get_device_info()
    return {
        "cpu": True,
        "cuda": "GPU" in device_info,
        "dml": False,
    }


@app.get("/speakers", response_model=list[Speaker], tags=["Speaker"])
async def get_speakers(
    model_size: str = Query(default=None, description="モデルサイズ (1.7B または 0.6B)"),
) -> list[dict]:
    """保存済みボイスモデルの一覧を返す。

    model_sizeを省略した場合はデフォルトモデルサイズを使用する。
    """
    size = model_size or _default_model_size
    if size not in ("1.7B", "0.6B"):
        raise HTTPException(
            status_code=400,
            detail=f"無効なモデルサイズ: {size}。1.7B または 0.6B を指定してください。",
        )
    return _build_speaker_list(size)


@app.get("/speaker_info", response_model=SpeakerInfo, tags=["Speaker"])
async def get_speaker_info(
    speaker_uuid: str = Query(..., description="話者UUID"),
    model_size: str = Query(default=None, description="モデルサイズ"),
) -> dict:
    """話者の詳細情報を返す。"""
    size = model_size or _default_model_size
    voices = list_voices_by_size(size)
    # speaker_uuidから話者を検索
    for idx, voice in enumerate(voices):
        if f"qwen-tts-{voice['nickname']}" == speaker_uuid:
            return {
                "policy": f"言語: {voice['language']} | モデルサイズ: {voice['model_size']}",
                "portrait": "",
                "style_infos": [
                    {
                        "id": idx,
                        "icon": "",
                        "portrait": "",
                        "voice_samples": [],
                    }
                ],
            }
    raise HTTPException(
        status_code=404,
        detail=f"話者UUID '{speaker_uuid}' が見つかりません。",
    )


@app.get("/supported_languages", tags=["System"])
async def get_supported_languages() -> list[str]:
    """サポートされている言語の一覧を返す。"""
    return SUPPORTED_LANGUAGES


@app.post("/audio_query", response_model=AudioQuery, tags=["Synthesis"])
async def create_audio_query(
    text: str = Query(..., description="合成するテキスト"),
    speaker: int = Query(..., description="話者ID"),
    model_size: str = Query(default=None, description="モデルサイズ (1.7B または 0.6B)"),
    language: str = Query(default=None, description="出力言語"),
    temperature: float = Query(default=0.65, description="感情値"),
    repetition_penalty: float = Query(default=1.15, description="繰り返し抑制係数"),
    top_p: float = Query(default=0.9, description="核サンプリング確率閾値"),
    top_k: int = Query(default=50, description="サンプリング候補数"),
) -> AudioQuery:
    """音声合成クエリを作成する。

    synthesisエンドポイントに渡すAudioQueryオブジェクトを返す。
    """
    size = model_size or _default_model_size
    if size not in ("1.7B", "0.6B"):
        raise HTTPException(
            status_code=400,
            detail=f"無効なモデルサイズ: {size}。1.7B または 0.6B を指定してください。",
        )
    # 話者IDの検証
    _speaker_id_to_nickname(speaker, size)

    # 言語の決定（省略時は話者の設定言語を使用）
    if language is None:
        voices = list_voices_by_size(size)
        lang = voices[speaker]["language"]
    else:
        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"サポートされていない言語: {language}。"
                f"サポート言語: {', '.join(SUPPORTED_LANGUAGES)}",
            )
        lang = language

    return AudioQuery(
        text=text,
        language=lang,
        model_size=size,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
    )


@app.post("/synthesis", tags=["Synthesis"])
async def synthesis(
    query: AudioQuery,
    speaker: int = Query(..., description="話者ID"),
    model_size: str = Query(default=None, description="モデルサイズ (1.7B または 0.6B)"),
    enable_interrogative_upspeak: bool = Query(
        default=True, description="疑問文の語尾を上げる（現在未使用）"
    ),
) -> Response:
    """音声合成を実行してWAVファイルを返す。

    audio_queryで作成したAudioQueryをボディに渡してください。
    """
    size = model_size or query.model_size or _default_model_size
    if size not in ("1.7B", "0.6B"):
        raise HTTPException(
            status_code=400,
            detail=f"無効なモデルサイズ: {size}。1.7B または 0.6B を指定してください。",
        )
    nickname = _speaker_id_to_nickname(speaker, size)
    if not query.text:
        raise HTTPException(status_code=400, detail="テキストが空です。")

    try:
        voice_prompt = load_voice(nickname, size)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    engine = get_engine()
    try:
        wav, sr = engine.generate_speech(
            text=query.text,
            language=query.language,
            voice_clone_prompt=voice_prompt,
            model_size=size,
            temperature=query.temperature,
            repetition_penalty=query.repetition_penalty,
            top_p=query.top_p,
            top_k=query.top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"音声合成に失敗しました: {e}"
        ) from e

    audio_bytes = _wav_to_bytes(wav, sr)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/multi_synthesis", tags=["Synthesis"])
async def multi_synthesis(
    queries: list[AudioQuery],
    speaker: int = Query(..., description="話者ID"),
    model_size: str = Query(default=None, description="モデルサイズ"),
) -> Response:
    """複数のAudioQueryを連結して音声合成し、1つのWAVファイルを返す。"""
    size = model_size or _default_model_size
    if size not in ("1.7B", "0.6B"):
        raise HTTPException(
            status_code=400,
            detail=f"無効なモデルサイズ: {size}。1.7B または 0.6B を指定してください。",
        )
    nickname = _speaker_id_to_nickname(speaker, size)

    try:
        voice_prompt = load_voice(nickname, size)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    engine = get_engine()
    wavs = []
    sr_final = DEFAULT_SAMPLE_RATE
    silence = np.zeros(int(sr_final * INTER_QUERY_SILENCE_DURATION), dtype=np.float32)

    for query in queries:
        if not query.text:
            continue
        try:
            wav, sr = engine.generate_speech(
                text=query.text,
                language=query.language,
                voice_clone_prompt=voice_prompt,
                model_size=size,
                temperature=query.temperature,
                repetition_penalty=query.repetition_penalty,
                top_p=query.top_p,
                top_k=query.top_k,
            )
            sr_final = sr
            wavs.append(wav)
            wavs.append(silence)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"音声合成に失敗しました: {e}"
            ) from e

    if not wavs:
        raise HTTPException(status_code=400, detail="合成するテキストがありません。")

    combined = np.concatenate(wavs[:-1])  # 末尾の無音を除く
    audio_bytes = _wav_to_bytes(combined, sr_final)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/connect_waves", tags=["Synthesis"])
async def connect_waves(waves: list[bytes]) -> Response:
    """複数のWAVバイト列を結合して1つのWAVファイルを返す。"""
    if not waves:
        raise HTTPException(status_code=400, detail="結合するWAVデータがありません。")

    wavs = []
    sr_final = DEFAULT_SAMPLE_RATE
    for wave_bytes in waves:
        wav, sr = sf.read(io.BytesIO(wave_bytes))
        sr_final = sr
        wavs.append(wav.astype(np.float32))

    combined = np.concatenate(wavs)
    audio_bytes = _wav_to_bytes(combined, sr_final)
    return Response(content=audio_bytes, media_type="audio/wav")


# --- ボイスモデル管理エンドポイント（拡張） ---


@app.get("/voice_models", tags=["VoiceModel"])
async def get_voice_models(
    model_size: str = Query(default=None, description="モデルサイズ"),
) -> list[dict]:
    """保存済みボイスモデルの一覧を返す（拡張API）。"""
    size = model_size or _default_model_size
    voices = list_voices_by_size(size)
    result = []
    for idx, voice in enumerate(voices):
        result.append(
            {
                "speaker_id": idx,
                "nickname": voice["nickname"],
                "language": voice["language"],
                "model_size": voice["model_size"],
                "created_at": voice["created_at"],
            }
        )
    return result


@app.post("/voice_models/upload", response_model=VoiceCreateResponse, tags=["VoiceModel"])
async def upload_voice_model(
    audio: UploadFile = File(..., description="リファレンス音声ファイル"),
    ref_text: str = Form(..., description="リファレンス音声の文字起こし"),
    nickname: str = Form(..., description="ボイスモデルのニックネーム"),
    language: str = Form(default="Japanese", description="リファレンス音声の言語"),
    model_size: str = Form(default=None, description="モデルサイズ"),
) -> VoiceCreateResponse:
    """リファレンス音声からボイスモデルを作成・保存する（拡張API）。"""
    size = model_size or _default_model_size
    if size not in ("1.7B", "0.6B"):
        raise HTTPException(
            status_code=400,
            detail=f"無効なモデルサイズ: {size}。1.7B または 0.6B を指定してください。",
        )
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"サポートされていない言語: {language}。"
            f"サポート言語: {', '.join(SUPPORTED_LANGUAGES)}",
        )
    if not nickname.strip():
        raise HTTPException(status_code=400, detail="ニックネームを入力してください。")

    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await audio.read())
        audio_path = f.name

    engine = get_engine()
    try:
        prompt_items = engine.create_voice_prompt(
            ref_audio=audio_path,
            ref_text=ref_text,
            model_size=size,
        )
        save_voice(nickname, prompt_items, language, size)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"ボイスモデルの作成に失敗しました: {e}"
        ) from e
    finally:
        Path(audio_path).unlink(missing_ok=True)

    voices = list_voices_by_size(size)
    speaker_id = next(
        (i for i, v in enumerate(voices) if v["nickname"] == nickname), 0
    )
    return VoiceCreateResponse(
        speaker_id=speaker_id,
        nickname=nickname,
        language=language,
        model_size=size,
    )


@app.delete("/voice_models/{nickname}", tags=["VoiceModel"])
async def delete_voice_model(
    nickname: str,
    model_size: str = Query(default=None, description="モデルサイズ"),
) -> dict:
    """ボイスモデルを削除する（拡張API）。"""
    size = model_size or _default_model_size
    voices = list_voices_by_size(size)
    if not any(v["nickname"] == nickname for v in voices):
        raise HTTPException(
            status_code=404,
            detail=f"ボイスモデル '{nickname}' が見つかりません（{size}）。",
        )
    remove_voice(nickname, size)
    return {"deleted": nickname, "model_size": size}


@app.get("/voice_models/{nickname}/export", tags=["VoiceModel"])
async def export_voice_model(
    nickname: str,
    model_size: str = Query(default=None, description="モデルサイズ"),
) -> Response:
    """ボイスモデルをダウンロードする（拡張API）。"""
    size = model_size or _default_model_size
    try:
        data, filename = export_voice(nickname, size)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post(
    "/voice_clone/synthesis",
    tags=["VoiceClone"],
)
async def voice_clone_synthesis(
    text: str = Form(..., description="合成するテキスト"),
    audio: UploadFile = File(..., description="リファレンス音声ファイル"),
    ref_text: str = Form(..., description="リファレンス音声の文字起こし"),
    language: str = Form(default="Japanese", description="出力言語"),
    model_size: str = Form(default=None, description="モデルサイズ"),
    temperature: float = Form(default=0.65, description="感情値"),
    repetition_penalty: float = Form(default=1.15, description="繰り返し抑制係数"),
    top_p: float = Form(default=0.9, description="核サンプリング確率閾値"),
    top_k: int = Form(default=50, description="サンプリング候補数"),
) -> Response:
    """リファレンス音声から直接音声合成を行う（ボイスモデル保存なし）（拡張API）。"""
    size = model_size or _default_model_size
    if size not in ("1.7B", "0.6B"):
        raise HTTPException(
            status_code=400,
            detail=f"無効なモデルサイズ: {size}。1.7B または 0.6B を指定してください。",
        )
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"サポートされていない言語: {language}。"
            f"サポート言語: {', '.join(SUPPORTED_LANGUAGES)}",
        )

    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await audio.read())
        audio_path = f.name

    engine = get_engine()
    try:
        wav, sr = engine.generate_speech_direct(
            text=text,
            language=language,
            ref_audio=audio_path,
            ref_text=ref_text,
            model_size=size,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"音声合成に失敗しました: {e}"
        ) from e
    finally:
        Path(audio_path).unlink(missing_ok=True)

    audio_bytes = _wav_to_bytes(wav, sr)
    return Response(content=audio_bytes, media_type="audio/wav")


# --- エントリポイント ---


def main():
    """APIサーバーを起動する。"""
    global _default_model_size

    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Web APIサーバー (VOICEVOX互換)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="バインドするホスト (デフォルト: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50021,
        help="バインドするポート (デフォルト: 50021)",
    )
    parser.add_argument(
        "--model-size",
        default="1.7B",
        choices=["1.7B", "0.6B"],
        help="使用するモデルサイズ (デフォルト: 1.7B)",
    )
    args = parser.parse_args()

    _default_model_size = args.model_size

    print("Qwen3-TTS Web API サーバーを起動します")
    print(f"  ホスト: {args.host}")
    print(f"  ポート: {args.port}")
    print(f"  モデルサイズ: {args.model_size}")
    print(f"  API ドキュメント: http://{args.host}:{args.port}/docs")
    print(f"  ReDoc: http://{args.host}:{args.port}/redoc")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
