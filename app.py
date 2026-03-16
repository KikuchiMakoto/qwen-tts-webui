"""Qwen3-TTS WebUI - Streamlit アプリケーション."""

import io
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import torch

from engine import GREETINGS, SUPPORTED_LANGUAGES, TTSEngine
from voice_store import (
    export_voice,
    import_voice,
    list_voices_by_size,
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


def show_completion_notification(message: str = "完了しました！"):
    """完了通知を表示する（Toast + 音声）."""
    st.toast(message, icon="✅")
    components.html(
        """
        <script>
        const audio = new Audio('/static/notify.ogg');
        audio.load();
        audio.play().catch(() => {
            console.log('音声再生がブロックされました');
        });
        </script>
        """,
        height=0,
    )


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
            "オリジナルボイスモデル学習",
            "音声合成",
            "オリジナル音声即時合成",
            "音声合成WebAPI",
        ],
        index=0,
    )

    st.divider()

    # 生成パラメータ設定（音声合成モードで使用）
    with st.expander("生成パラメータ設定", expanded=False):
        gen_temperature = st.slider(
            "温度（感情値）",
            min_value=0.30,
            max_value=1.30,
            value=0.65,
            step=0.05,
            key="gen_temperature",
            help="感情の豊かさを制御します。高いほど表現豊かでランダムな発音、低いほど落ち着いた安定した発音になります。",
        )
        gen_top_p = st.slider(
            "Top-p（核サンプリング）",
            min_value=0.80,
            max_value=1.00,
            value=0.90,
            step=0.05,
            key="gen_top_p",
            help="累積確率が上位p以内のトークンのみからサンプリングします。低いほど安定した出力になります。",
        )
        gen_top_k = st.slider(
            "Top-k",
            min_value=10,
            max_value=50,
            value=50,
            step=1,
            key="gen_top_k",
            help="確率の高い上位k件のトークンのみからサンプリングします。低いほど安定した出力になります。",
        )
        gen_repetition_penalty = st.slider(
            "繰り返し抑制（Repetition Penalty）",
            min_value=1.00,
            max_value=1.50,
            value=1.15,
            step=0.05,
            key="gen_repetition_penalty",
            help="同じ音やフレーズの繰り返しを抑制します。高いほど繰り返しが起きにくくなります。",
        )

    st.divider()

    # モードに応じた履歴クリアボタン
    if mode == "音声合成" and st.session_state.tts_history:
        if st.button("チャット履歴クリア", key="clear_tts"):
            st.session_state.tts_history = []
            st.rerun()
    elif mode == "オリジナル音声即時合成" and st.session_state.instant_history:
        if st.button("チャット履歴クリア", key="clear_instant"):
            st.session_state.instant_history = []
            st.rerun()


# =============================================================================
# モード1: オリジナルボイスモデル学習
# =============================================================================
if mode == "オリジナルボイスモデル学習":
    st.header("オリジナルボイスモデル学習")
    st.markdown("リファレンス音声からオリジナルボイスモデルを作成します。")

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
                st.error(
                    "音声ファイル、文字起こし、ニックネームをすべて入力してください。"
                )
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
                show_completion_notification("ボイスモデル作成完了")
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
                show_completion_notification("プレビュー完了")
                st.audio(audio_to_bytes(wav, sr), format="audio/wav")

    with col_saved:
        st.subheader("保存済みモデル")
        voices = list_voices_by_size(model_size)

        if not voices:
            st.info(f"保存されたボイスモデルはありません（{model_size}）。")

        for voice in voices:
            with st.container(border=True):
                st.markdown(f"**{voice['nickname']}**")
                st.caption(
                    f"言語: {voice['language']} | サイズ: {voice['model_size']} | "
                    f"作成日: {voice['created_at'][:10]}"
                )

                c1, c2 = st.columns(2)
                with c1:
                    data, filename = export_voice(
                        voice["nickname"], voice["model_size"]
                    )
                    st.download_button(
                        "エクスポート",
                        data=data,
                        file_name=filename,
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
                        remove_voice(voice["nickname"], voice["model_size"])
                        st.rerun()

        st.divider()
        st.subheader("モデルインポート")

        import_file = st.file_uploader(
            f"ボイスモデルファイル (.pt) - {model_size}向けのみ",
            type=["pt"],
            key="import_file",
        )
        import_nickname = st.text_input("インポート名", key="import_nickname")
        import_lang = st.selectbox("言語", SUPPORTED_LANGUAGES, key="import_lang")

        if st.button("インポート", use_container_width=True):
            if import_file and import_nickname:
                try:
                    import_voice(
                        import_nickname, import_file.read(), import_lang, model_size
                    )
                    st.success(f"'{import_nickname}' をインポートしました。")
                    st.rerun()
                except ValueError as e:
                    st.error(f"インポート失敗: {e}")
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
        voices = list_voices_by_size(model_size)
        voice_options = [v["nickname"] for v in voices]
        if voice_options:
            selected_voice = st.selectbox(
                "ボイスモデル", voice_options, key="tts_voice"
            )
            if selected_voice:
                voice_prompt = load_voice(selected_voice, model_size)
        else:
            st.warning(
                f"保存されたボイスモデルがありません（{model_size}）。"
                "「オリジナルボイスモデル学習」モードで先にモデルを作成してください。"
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
                        temperature=gen_temperature,
                        repetition_penalty=gen_repetition_penalty,
                        top_p=gen_top_p,
                        top_k=gen_top_k,
                    )
                audio_bytes = audio_to_bytes(wav, sr)
                msg_id = str(uuid.uuid4())
                st.markdown("音声を生成しました")
                show_completion_notification("音声合成完了")
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
# モード3: オリジナル音声即時合成
# =============================================================================
elif mode == "オリジナル音声即時合成":
    st.header("オリジナル音声即時合成")
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
                        temperature=gen_temperature,
                        repetition_penalty=gen_repetition_penalty,
                        top_p=gen_top_p,
                        top_k=gen_top_k,
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


# =============================================================================
# モード4: 音声合成WebAPI
# =============================================================================
elif mode == "音声合成WebAPI":
    st.header("音声合成WebAPI")
    st.markdown(
        "VOICEVOX互換のHTTP APIサーバーとして利用できます。\n"
        "以下のコマンドでAPIサーバーを起動し、他のアプリケーションから音声合成を呼び出せます。"
    )

    st.subheader("サーバー起動コマンド")

    col_cmd1, col_cmd2 = st.columns(2)
    with col_cmd1:
        st.code("uv run python api_server.py", language="bash")
        st.caption("デフォルト: host=127.0.0.1, port=50021, model=1.7B")

    with col_cmd2:
        st.code(
            "uv run python api_server.py --host 0.0.0.0 --port 50021 --model-size 0.6B",
            language="bash",
        )
        st.caption("ネットワーク公開 + 0.6Bモデル使用の例")

    st.divider()

    st.subheader("主要APIエンドポイント（VOICEVOX互換）")

    with st.expander("GET /version - バージョン取得", expanded=False):
        st.code('curl http://127.0.0.1:50021/version', language="bash")

    with st.expander("GET /speakers - 話者（ボイスモデル）一覧", expanded=False):
        st.code('curl http://127.0.0.1:50021/speakers', language="bash")
        st.caption(
            "保存済みボイスモデルが話者として返されます。"
            "speaker_id は /voice_models で確認できます。"
        )

    with st.expander(
        "POST /audio_query + POST /synthesis - 音声合成（2ステップ）", expanded=True
    ):
        st.markdown("**ステップ1: AudioQueryを作成**")
        st.code(
            'curl -X POST "http://127.0.0.1:50021/audio_query?text=こんにちは&speaker=0"'
            " > query.json",
            language="bash",
        )
        st.markdown("**ステップ2: 音声合成**")
        st.code(
            'curl -X POST "http://127.0.0.1:50021/synthesis?speaker=0"'
            " -H 'Content-Type: application/json'"
            " -d @query.json"
            " --output output.wav",
            language="bash",
        )

    st.divider()

    st.subheader("拡張APIエンドポイント")

    with st.expander("POST /voice_models/upload - ボイスモデル作成", expanded=False):
        st.code(
            "curl -X POST http://127.0.0.1:50021/voice_models/upload \\\n"
            '  -F "audio=@reference.wav" \\\n'
            '  -F "ref_text=こんにちは、はじめまして。" \\\n'
            '  -F "nickname=myvoice" \\\n'
            '  -F "language=Japanese"',
            language="bash",
        )

    with st.expander(
        "POST /voice_clone/synthesis - リファレンス音声から直接合成", expanded=False
    ):
        st.code(
            "curl -X POST http://127.0.0.1:50021/voice_clone/synthesis \\\n"
            '  -F "text=合成するテキスト" \\\n'
            '  -F "audio=@reference.wav" \\\n'
            '  -F "ref_text=リファレンス音声の文字起こし" \\\n'
            '  -F "language=Japanese" \\\n'
            "  --output output.wav",
            language="bash",
        )

    with st.expander("GET /voice_models - ボイスモデル一覧（拡張）", expanded=False):
        st.code("curl http://127.0.0.1:50021/voice_models", language="bash")

    st.divider()

    st.subheader("APIドキュメント")
    st.markdown(
        "サーバー起動後、以下のURLでインタラクティブなAPIドキュメントを参照できます:\n\n"
        "- **Swagger UI**: http://127.0.0.1:50021/docs\n"
        "- **ReDoc**: http://127.0.0.1:50021/redoc"
    )

    st.divider()

    st.subheader("Pythonサンプルコード")
    st.code(
        '''import requests

BASE_URL = "http://127.0.0.1:50021"

# ステップ1: AudioQueryを作成
response = requests.post(
    f"{BASE_URL}/audio_query",
    params={"text": "こんにちは", "speaker": 0},
)
query = response.json()

# ステップ2: 音声合成
response = requests.post(
    f"{BASE_URL}/synthesis",
    params={"speaker": 0},
    json=query,
)

# WAVファイルとして保存
with open("output.wav", "wb") as f:
    f.write(response.content)
''',
        language="python",
    )
