"""Qwen3-TTS WebUI - Streamlit アプリケーション."""

import io
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch

from engine import GREETINGS, SUPPORTED_LANGUAGES, TTSEngine
from voice_store import (
    export_voice,
    import_voice,
    list_voices,
    load_voice,
    remove_voice,
    save_voice,
)

st.set_page_config(
    page_title="Qwen3-TTS WebUI",
    layout="wide",
)


# --- ユーティリティ ---


def audio_to_bytes(wav: np.ndarray, sr: int) -> bytes:
    """numpy音声をWAVバイト列に変換する。"""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def save_uploaded_audio(uploaded_file) -> str:
    """アップロードされた音声をテンポラリファイルに保存しパスを返す。"""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.getvalue())
        return f.name


# --- セッション状態の初期化 ---

if "engine" not in st.session_state:
    st.session_state.engine = TTSEngine()
if "tts_history" not in st.session_state:
    st.session_state.tts_history = []
if "instant_history" not in st.session_state:
    st.session_state.instant_history = []


# --- サイドバー ---

with st.sidebar:
    st.title("Qwen3-TTS WebUI")
    st.caption(f"デバイス: {st.session_state.engine.get_device_info()}")

    model_size = st.selectbox(
        "モデルサイズ",
        ["1.7B", "0.6B"],
        index=0,
        help="1.7B（デフォルト）は高品質、0.6Bは軽量版です。",
    )

    st.divider()

    mode = st.radio(
        "モード選択",
        [
            "カスタムボイスモデル学習",
            "音声合成",
            "カスタム音声即時合成",
        ],
        index=0,
    )

    st.divider()

    # モードに応じた履歴クリアボタン
    if mode == "音声合成" and st.session_state.tts_history:
        if st.button("チャット履歴クリア", key="clear_tts"):
            st.session_state.tts_history = []
            st.rerun()
    elif mode == "カスタム音声即時合成" and st.session_state.instant_history:
        if st.button("チャット履歴クリア", key="clear_instant"):
            st.session_state.instant_history = []
            st.rerun()


# =============================================================================
# モード1: カスタムボイスモデル学習
# =============================================================================
if mode == "カスタムボイスモデル学習":
    st.header("カスタムボイスモデル学習")
    st.markdown("リファレンス音声からカスタムボイスモデルを作成します。")

    col_create, col_saved = st.columns([3, 2])

    with col_create:
        st.subheader("新規モデル作成")

        audio_file = st.file_uploader(
            "リファレンス音声ファイル",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="train_audio",
            help="3秒以上の音声ファイルを推奨します。",
        )

        ref_text = st.text_area(
            "音声の文字起こし",
            placeholder="アップロードした音声の内容をテキストで入力してください",
            key="train_text",
        )

        ref_lang = st.selectbox(
            "リファレンス音声の言語",
            SUPPORTED_LANGUAGES,
            index=0,
            key="train_lang",
        )

        nickname = st.text_input(
            "ニックネーム",
            placeholder="ボイスモデルの名前を入力",
            key="train_nickname",
        )

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            create_clicked = st.button(
                "モデル作成", type="primary", use_container_width=True
            )

        with col_btn2:
            preview_clicked = st.button("挨拶プレビュー", use_container_width=True)

        if create_clicked:
            if not audio_file or not ref_text or not nickname:
                st.error("音声ファイル、文字起こし、ニックネームをすべて入力してください。")
            else:
                audio_path = save_uploaded_audio(audio_file)
                with st.spinner("ボイスモデルを作成中..."):
                    prompt_items = st.session_state.engine.create_voice_prompt(
                        ref_audio=audio_path,
                        ref_text=ref_text,
                        model_size=model_size,
                    )
                    save_voice(nickname, prompt_items, ref_lang, model_size)
                st.success(f"ボイスモデル '{nickname}' を作成しました。")
                st.rerun()

        if preview_clicked:
            if not audio_file or not ref_text:
                st.error("音声ファイルと文字起こしを入力してください。")
            else:
                audio_path = save_uploaded_audio(audio_file)
                greeting = GREETINGS.get(ref_lang, GREETINGS["English"])
                with st.spinner(f"挨拶音声を生成中（{ref_lang}）..."):
                    prompt_items = st.session_state.engine.create_voice_prompt(
                        ref_audio=audio_path,
                        ref_text=ref_text,
                        model_size=model_size,
                    )
                    wav, sr = st.session_state.engine.generate_speech(
                        text=greeting,
                        language=ref_lang,
                        voice_clone_prompt=prompt_items,
                        model_size=model_size,
                    )
                st.info(f"挨拶: {greeting}")
                st.audio(audio_to_bytes(wav, sr), format="audio/wav")

    with col_saved:
        st.subheader("保存済みモデル")
        voices = list_voices()

        if not voices:
            st.info("保存されたボイスモデルはありません。")

        for voice in voices:
            with st.container(border=True):
                st.markdown(f"**{voice['nickname']}**")
                st.caption(
                    f"言語: {voice['language']} | サイズ: {voice['model_size']} | "
                    f"作成日: {voice['created_at'][:10]}"
                )

                c1, c2 = st.columns(2)
                with c1:
                    data = export_voice(voice["nickname"])
                    st.download_button(
                        "エクスポート",
                        data=data,
                        file_name=f"{voice['nickname']}.pt",
                        mime="application/octet-stream",
                        use_container_width=True,
                        key=f"export_{voice['nickname']}",
                    )
                with c2:
                    if st.button(
                        "削除",
                        key=f"del_{voice['nickname']}",
                        use_container_width=True,
                    ):
                        remove_voice(voice["nickname"])
                        st.rerun()

        st.divider()
        st.subheader("モデルインポート")

        import_file = st.file_uploader(
            "ボイスモデルファイル (.pt)",
            type=["pt"],
            key="import_file",
        )
        import_nickname = st.text_input("インポート名", key="import_nickname")
        import_lang = st.selectbox("言語", SUPPORTED_LANGUAGES, key="import_lang")

        if st.button("インポート", use_container_width=True):
            if import_file and import_nickname:
                import_voice(
                    import_nickname, import_file.read(), import_lang, model_size
                )
                st.success(f"'{import_nickname}' をインポートしました。")
                st.rerun()
            else:
                st.error("ファイルとインポート名を入力してください。")


# =============================================================================
# モード2: 音声合成
# =============================================================================
elif mode == "音声合成":
    st.header("音声合成")

    # 設定エリア
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        voice_source = st.radio(
            "ボイスモデルソース",
            ["保存済みモデル", "ファイルアップロード"],
            horizontal=True,
            key="tts_source",
        )

    with col_s2:
        output_lang = st.selectbox(
            "出力言語",
            SUPPORTED_LANGUAGES,
            index=0,
            key="tts_lang",
        )

    # ボイスモデルの読み込み
    voice_prompt = None

    if voice_source == "保存済みモデル":
        voices = list_voices()
        voice_options = [v["nickname"] for v in voices]
        if voice_options:
            selected_voice = st.selectbox(
                "ボイスモデル", voice_options, key="tts_voice"
            )
            if selected_voice:
                voice_prompt = load_voice(selected_voice)
        else:
            st.warning(
                "保存されたボイスモデルがありません。"
                "「カスタムボイスモデル学習」モードで先にモデルを作成してください。"
            )
    else:
        uploaded_model = st.file_uploader(
            "ボイスモデルファイル (.pt)",
            type=["pt"],
            key="tts_upload_model",
        )
        if uploaded_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
                f.write(uploaded_model.read())
                voice_prompt = torch.load(f.name, weights_only=False)

    st.divider()

    # チャットインターフェース
    for msg in st.session_state.tts_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/wav")
                st.download_button(
                    "ダウンロード",
                    data=msg["audio"],
                    file_name=f"tts_{msg['id']}.wav",
                    mime="audio/wav",
                    key=f"dl_{msg['id']}",
                )

    if text_input := st.chat_input("合成するテキストを入力してください"):
        if voice_prompt is None:
            st.error("ボイスモデルを選択またはアップロードしてください。")
        else:
            # ユーザーメッセージ
            user_msg = {
                "role": "user",
                "content": text_input,
                "id": str(uuid.uuid4()),
            }
            st.session_state.tts_history.append(user_msg)

            with st.chat_message("user"):
                st.markdown(text_input)

            # 音声生成
            with st.chat_message("assistant"):
                with st.spinner("音声を生成中..."):
                    wav, sr = st.session_state.engine.generate_speech(
                        text=text_input,
                        language=output_lang,
                        voice_clone_prompt=voice_prompt,
                        model_size=model_size,
                    )
                audio_bytes = audio_to_bytes(wav, sr)
                msg_id = str(uuid.uuid4())
                st.markdown("音声を生成しました")
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    "ダウンロード",
                    data=audio_bytes,
                    file_name=f"tts_{msg_id}.wav",
                    mime="audio/wav",
                    key=f"dl_{msg_id}",
                )

            st.session_state.tts_history.append(
                {
                    "role": "assistant",
                    "content": "音声を生成しました",
                    "audio": audio_bytes,
                    "id": msg_id,
                }
            )


# =============================================================================
# モード3: カスタム音声即時合成
# =============================================================================
elif mode == "カスタム音声即時合成":
    st.header("カスタム音声即時合成")
    st.markdown(
        "リファレンス音声から直接音声合成を行います。ボイスモデルはキャッシュされません。"
    )

    col_ref, col_out = st.columns(2)

    with col_ref:
        st.subheader("リファレンス設定")
        instant_audio = st.file_uploader(
            "リファレンス音声",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="instant_audio",
            help="3秒以上の音声ファイルを推奨します。",
        )
        instant_ref_text = st.text_area(
            "文字起こし",
            placeholder="リファレンス音声の内容",
            key="instant_ref_text",
        )
        instant_ref_lang = st.selectbox(
            "リファレンス言語",
            SUPPORTED_LANGUAGES,
            index=0,
            key="instant_ref_lang",
        )

    with col_out:
        st.subheader("出力設定")
        instant_out_lang = st.selectbox(
            "出力言語",
            SUPPORTED_LANGUAGES,
            index=0,
            key="instant_out_lang",
        )

    st.divider()

    # チャットインターフェース
    for msg in st.session_state.instant_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/wav")
                st.download_button(
                    "ダウンロード",
                    data=msg["audio"],
                    file_name=f"instant_{msg['id']}.wav",
                    mime="audio/wav",
                    key=f"instant_dl_{msg['id']}",
                )

    if text_input := st.chat_input("合成するテキストを入力"):
        if not instant_audio or not instant_ref_text:
            st.error("リファレンス音声と文字起こしを入力してください。")
        else:
            # ユーザーメッセージ
            user_msg = {
                "role": "user",
                "content": text_input,
                "id": str(uuid.uuid4()),
            }
            st.session_state.instant_history.append(user_msg)

            with st.chat_message("user"):
                st.markdown(text_input)

            # 音声生成
            with st.chat_message("assistant"):
                audio_path = save_uploaded_audio(instant_audio)
                with st.spinner("リファレンス音声を解析して音声を生成中..."):
                    wav, sr = st.session_state.engine.generate_speech_direct(
                        text=text_input,
                        language=instant_out_lang,
                        ref_audio=audio_path,
                        ref_text=instant_ref_text,
                        model_size=model_size,
                    )
                audio_bytes = audio_to_bytes(wav, sr)
                msg_id = str(uuid.uuid4())
                st.markdown("音声を生成しました")
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    "ダウンロード",
                    data=audio_bytes,
                    file_name=f"instant_{msg_id}.wav",
                    mime="audio/wav",
                    key=f"instant_dl_{msg_id}",
                )

            st.session_state.instant_history.append(
                {
                    "role": "assistant",
                    "content": "音声を生成しました",
                    "audio": audio_bytes,
                    "id": msg_id,
                }
            )
