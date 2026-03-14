"""Qwen3-TTS Engine - モデル管理とTTS生成."""

import os

import numpy as np
import torch

# Qwen3-TTS がサポートする言語
SUPPORTED_LANGUAGES = [
    "Japanese",
    "Chinese",
    "English",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# 各言語の挨拶テキスト（プレビュー用）
GREETINGS = {
    "Japanese": "こんにちは、はじめまして。よろしくお願いします。",
    "Chinese": "你好，很高兴认识你。请多多关照。",
    "English": "Hello, nice to meet you. How are you doing today?",
    "Korean": "안녕하세요, 만나서 반갑습니다. 잘 부탁드립니다.",
    "German": "Hallo, freut mich, Sie kennenzulernen. Wie geht es Ihnen?",
    "French": "Bonjour, enchanté de vous rencontrer. Comment allez-vous?",
    "Russian": "Здравствуйте, приятно познакомиться. Как у вас дела?",
    "Portuguese": "Olá, prazer em conhecê-lo. Como você está?",
    "Spanish": "Hola, mucho gusto en conocerte. ¿Cómo estás?",
    "Italian": "Ciao, piacere di conoscerti. Come stai?",
}

# モデルID
MODEL_IDS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}


def get_device() -> str:
    """利用可能なデバイスを検出する。CUDA > CPU の優先順位。

    環境変数 FORCE_CPU=1 でCPUを強制できる。
    """
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    # ROCm環境: torch.cuda.is_available() はROCm版PyTorchでもTrueを返すため、
    # ROCm用PyTorch (pip install torch --index-url https://download.pytorch.org/whl/rocm6.3)
    # をインストールすれば自動的にGPUが使用される。
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """デバイスに適したdtypeを返す。"""
    if "cuda" in device:
        return torch.bfloat16
    return torch.float32


def get_attn_implementation(device: str) -> str:
    """利用可能なattention実装を返す。"""
    if "cuda" in device:
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    return "sdpa"


class TTSEngine:
    """Qwen3-TTS モデルのラッパー。モデルの遅延読み込みとキャッシュを行う。"""

    def __init__(self):
        self._model = None
        self._model_size: str | None = None
        self._device: str | None = None

    def _ensure_model(self, model_size: str = "1.7B"):
        """必要に応じてモデルをロードする。サイズが変わった場合は再ロードする。"""
        if self._model is not None and self._model_size == model_size:
            return

        from qwen_tts import Qwen3TTSModel

        device = get_device()
        dtype = get_dtype(device)
        attn_impl = get_attn_implementation(device)
        model_id = MODEL_IDS[model_size]

        self._model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        self._model_size = model_size
        self._device = device

    def create_voice_prompt(
        self,
        ref_audio,
        ref_text: str,
        model_size: str = "1.7B",
    ):
        """リファレンス音声から再利用可能なボイスクローンプロンプトを作成する。

        Args:
            ref_audio: ファイルパス、URL、base64文字列、または (numpy_array, sr) タプル
            ref_text: リファレンス音声の文字起こしテキスト
            model_size: 使用するモデルサイズ ("1.7B" or "0.6B")

        Returns:
            voice_clone_prompt: 再利用可能なプロンプトオブジェクト
        """
        self._ensure_model(model_size)
        prompt_items = self._model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return prompt_items

    def generate_speech(
        self,
        text: str,
        language: str,
        voice_clone_prompt,
        model_size: str = "1.7B",
        temperature: float = 0.65,
        repetition_penalty: float = 1.15,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> tuple[np.ndarray, int]:
        """保存済みボイスクローンプロンプトを使用して音声を合成する。

        Args:
            text: 合成するテキスト
            language: 出力言語
            voice_clone_prompt: create_voice_prompt で作成したプロンプト
            model_size: 使用するモデルサイズ
            temperature: 感情値。高いほど表現豊か、低いほど安定した発音になる。
            repetition_penalty: 繰り返し抑制係数。
            top_p: 核サンプリングの累積確率閾値。
            top_k: サンプリング候補数の上限。

        Returns:
            (wav, sample_rate) タプル
        """
        self._ensure_model(model_size)
        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
        )
        return wavs[0], sr

    def generate_speech_direct(
        self,
        text: str,
        language: str,
        ref_audio,
        ref_text: str,
        model_size: str = "1.7B",
        temperature: float = 0.65,
        repetition_penalty: float = 1.15,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> tuple[np.ndarray, int]:
        """リファレンス音声から直接音声を合成する（プロンプト未保存）。

        Args:
            text: 合成するテキスト
            language: 出力言語
            ref_audio: リファレンス音声のファイルパス
            ref_text: リファレンス音声の文字起こし
            model_size: 使用するモデルサイズ
            temperature: 感情値。高いほど表現豊か、低いほど安定した発音になる。
            repetition_penalty: 繰り返し抑制係数。
            top_p: 核サンプリングの累積確率閾値。
            top_k: サンプリング候補数の上限。

        Returns:
            (wav, sample_rate) タプル
        """
        self._ensure_model(model_size)
        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
        )
        return wavs[0], sr

    def get_device_info(self) -> str:
        """現在のデバイス情報を返す。"""
        device = get_device()
        if "cuda" in device:
            name = torch.cuda.get_device_name(0)
            return f"GPU: {name}"
        return "CPU"
