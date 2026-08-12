import os
import sys
import logging
import random
import tempfile
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile
import re
import uuid

temp_audio_dir = "./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# OmniVoice is installed as a pip package (official k2-fsa/OmniVoice >= 0.2.0)
# https://github.com/k2-fsa/OmniVoice
# ---------------------------------------------------------------------------
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

from subtitle import subtitle_maker

# Attempt to import Whisper's supported language dict to filter unsupported languages
try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Device Selection
#
# Colab hands out a CPU runtime by default, and forgetting to switch to a GPU
# is the most common way this app "breaks". Detect it up front and say exactly
# what to do, instead of failing deep inside from_pretrained with a stack trace.
# Set OMNIVOICE_DEVICE to override (e.g. "cpu" to force it).
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("OMNIVOICE_DEVICE", "").strip().lower()
if not DEVICE:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda" and not torch.cuda.is_available():
    print("⚠️  OMNIVOICE_DEVICE=cuda was requested but no GPU is visible — using CPU.")
    DEVICE = "cpu"

if DEVICE == "cpu":
    print(
        "\n" + "=" * 72 + "\n"
        "⚠️  NO GPU DETECTED — the app still runs, but expect several minutes\n"
        "    per clip instead of a few seconds.\n\n"
        "    In Colab:  Runtime → Change runtime type → T4 GPU → Save,\n"
        "               then re-run the cells.\n"
        + "=" * 72 + "\n"
    )

# float16 is a GPU optimisation; on CPU those kernels are missing or slower
# than plain float32.
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# Model Loading (Global Scope)
#
# Downloads from Hugging Face. Set HF_TOKEN for full download speed —
# anonymous downloads are throttled/blocked (the Colab notebook sets it
# from the HF_TOKEN form field).
# ---------------------------------------------------------------------------
print(f"Loading model from k2-fsa/OmniVoice to {DEVICE} ...")

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=DEVICE,
    dtype=DTYPE,
    load_asr=False,
)
sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Event Tags & JS Functions
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]",
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]",
    "[dissatisfaction-hnn]"
]

# JS for Voice Clone Tab Textbox
INSERT_TAG_JS_VC = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vc_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

# JS for Voice Design Tab Textbox
INSERT_TAG_JS_VD = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vd_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

# ---------------------------------------------------------------------------
# UI Configurations & Language Mappings
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
    "English Accent": [
        "American Accent", "Australian Accent", "British Accent", "Chinese Accent",
        "Canadian Accent", "Indian Accent", "Korean Accent", "Portuguese Accent",
        "Russian Accent", "Japanese Accent"
    ],
    "Chinese Dialect": [
        "Henan Dialect", "Shaanxi Dialect", "Sichuan Dialect", "Guizhou Dialect",
        "Yunnan Dialect", "Guilin Dialect", "Jinan Dialect", "Shijiazhuang Dialect",
        "Gansu Dialect", "Ningxia Dialect", "Qingdao Dialect", "Northeast Dialect"
    ],
}

DIALECT_MAP = {
    "Henan Dialect": "河南话", "Shaanxi Dialect": "陕西话", "Sichuan Dialect": "四川话",
    "Guizhou Dialect": "贵州话", "Yunnan Dialect": "云南话", "Guilin Dialect": "桂林话",
    "Jinan Dialect": "济南话", "Shijiazhuang Dialect": "石家庄话", "Gansu Dialect": "甘肃话",
    "Ningxia Dialect": "宁夏话", "Qingdao Dialect": "青岛话", "Northeast Dialect": "东北话",
}

_ATTR_INFO = {
    "English Accent": "Only effective for English speech.",
    "Chinese Dialect": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Core Logic & Helpers
# ---------------------------------------------------------------------------
_WHISPER_BY_NAME = {str(k).lower(): v for k, v in (WHISPER_LANGUAGE_CODE or {}).items()}
_WHISPER_BY_CODE = {str(v).lower(): v for v in (WHISPER_LANGUAGE_CODE or {}).values()}


def whisper_lang_code(lang):
    """Map an OmniVoice display name to a Whisper language code, or None.

    OmniVoice ships 646 language names, Whisper handles 84, and the two lists
    disagree on naming: OmniVoice lists regional varieties ("Egyptian Arabic",
    "Northern Pashto") where Whisper has a single umbrella entry. So fall back
    to matching whole words of 4+ characters, which resolves those varieties to
    their parent language without letting a short code like "ar" match inside
    an unrelated name such as "Western Maninkakan" — the substring test this
    replaced matched almost everything, so nothing was ever filtered out.
    """
    if not lang or str(lang).strip().lower() in ("", "auto"):
        return None

    key = str(lang).strip().lower()
    if key in _WHISPER_BY_NAME:
        return _WHISPER_BY_NAME[key]
    if key in _WHISPER_BY_CODE:
        return _WHISPER_BY_CODE[key]

    tokens = {t for t in re.findall(r"[a-z]+", key) if len(t) >= 4}
    if tokens:
        for name, code in _WHISPER_BY_NAME.items():
            if tokens & set(re.findall(r"[a-z]+", name)):
                return code
    return None


def _is_whisper_supported(lang):
    """Check if the selected language is supported by Whisper to save processing time."""
    if not lang or lang == "Auto" or WHISPER_LANGUAGE_CODE is None:
        return True

    return whisper_lang_code(lang) is not None

def generate_subtitles_if_needed(wav_path, lang, want_subs):
    """Generates Subtitles only if user requested them and language is supported."""
    if not want_subs:
        return None, None, None

    if not _is_whisper_supported(lang):
        logging.warning(f"Language '{lang}' is likely unsupported by Whisper. Skipping subtitle generation.")
        return None, None, None

    try:
        whisper_results = subtitle_maker(wav_path, whisper_lang_code(lang))
        if whisper_results and len(whisper_results) > 3:
            return whisper_results[1], whisper_results[2], whisper_results[3]
    except Exception as e:
        logging.warning(f"Subtitle generation failed: {e}")

    return None, None, None


def tts_file_name(text, language="en"):
    global temp_audio_dir

    # --- Clean text ---
    # \w keeps letters in every script, so Chinese/Hindi/Sinhala text still
    # produces a readable stem instead of being stripped down to "audio".
    clean_text = re.sub(r'[^\w\s-]', '', text, flags=re.UNICODE)
    clean_text = re.sub(r'[\s-]+', '_', clean_text).strip('_').lower()

    if not clean_text:
        clean_text = "audio"

    # --- Truncate ---
    truncated = clean_text[:20]

    # --- Clean language ---
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"
    lang = re.sub(r'[^\w_]', '', lang, flags=re.UNICODE) or "unknown"

    # --- Random suffix ---
    rand = uuid.uuid4().hex[:8].upper()

    # --- Final filename ---
    return f"{temp_audio_dir}/{truncated}_{lang}_{rand}.wav"


def apply_seed(seed):
    """Seed every RNG generation draws from, and report the seed actually used.

    Zero-shot TTS is stochastic, so the same text and settings give a slightly
    different take each time. A seed below zero means "pick one for me" — we
    still return it so the status line can show what to type in to get that
    exact take back.
    """
    try:
        resolved = int(seed)
    except (TypeError, ValueError):
        resolved = -1

    if resolved < 0:
        resolved = random.randint(0, 2**31 - 1)

    random.seed(resolved)
    np.random.seed(resolved % (2**32))
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)

    return resolved


def _sentence_split(text, lang_code=None):
    """Split text into sentences, falling back to punctuation if sentencex can't."""
    try:
        from sentencex import segment
        sentences = [s.strip() for s in segment(lang_code or "en", text) if s and s.strip()]
        if sentences:
            return sentences
    except Exception as e:
        logging.warning(f"sentencex unavailable ({e}); using punctuation fallback.")

    return [s.strip() for s in re.split(r'(?<=[.!?。！？])\s+', text) if s.strip()]


def _hard_wrap(sentence, limit):
    """Last resort for a single sentence longer than the whole chunk budget."""
    words = sentence.split()
    if len(words) < 2:
        # Scripts written without spaces (Chinese, Japanese, Thai, ...).
        return [sentence[i:i + limit] for i in range(0, len(sentence), limit)]

    lines, current = [], ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def split_into_chunks(text, limit, lang_code=None):
    """Group whole sentences into chunks of at most `limit` characters.

    Long text handed to the model in one call gets truncated or drifts, so
    split on sentence boundaries — never mid-sentence, and never inside an
    event tag like [laughter] — and stitch the audio back together afterwards.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks, current = [], ""
    for sentence in _sentence_split(text, lang_code):
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    # A single sentence can still blow the budget; allow some overshoot before
    # resorting to a mid-sentence break, which is always audible.
    wrapped = []
    for chunk in chunks:
        if len(chunk) <= limit * 1.5:
            wrapped.append(chunk)
        else:
            wrapped.extend(_hard_wrap(chunk, limit))
    return wrapped


def _voice_lock_prompt(waveform, text):
    """Clone the voice out of audio we just generated, to reuse for later chunks.

    Voice Design draws a fresh random voice on every call, so without this
    chunk 2 of a long script would not sound like chunk 1.
    """
    try:
        path = os.path.join(temp_audio_dir, f"_voicelock_{uuid.uuid4().hex[:8]}.wav")
        wavfile.write(path, sampling_rate, (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16))
        return model.create_voice_clone_prompt(ref_audio=path, ref_text=text)
    except Exception as e:
        logging.warning(f"Could not lock the voice across chunks: {e}")
        return None


def _gen_core(
    text, language, ref_audio, instruct, num_step, guidance_scale,
    denoise, speed, duration, preprocess_prompt, postprocess_output, mode, ref_text=None,
    seed=-1, auto_split=True, chunk_chars=300, chunk_pause=0.35
):
    """Core Text-to-Speech Generation Logic"""
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    text = text.strip()

    if mode == "clone" and not ref_audio:
        return None, "Please upload a reference audio."

    if mode == "clone" and ref_audio and not ref_text:
        try:
            whisper_results = subtitle_maker(ref_audio, whisper_lang_code(language))
            if whisper_results and len(whisper_results) > 7:
                ref_text = whisper_results[7]
        except Exception as e:
            logging.warning(f"Fallback transcription failed: {e}")

    used_seed = apply_seed(seed)

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None
    base_kw: Dict[str, Any] = dict(language=lang, generation_config=gen_config)

    if speed is not None and float(speed) != 1.0:
        base_kw["speed"] = float(speed)

    # A fixed duration describes one utterance, so it cannot be spread across
    # several chunks — those runs stay single-shot.
    fixed_duration = float(duration) if duration is not None and float(duration) > 0 else None
    if fixed_duration is not None:
        base_kw["duration"] = fixed_duration

    if auto_split and fixed_duration is None:
        chunks = split_into_chunks(text, max(60, int(chunk_chars or 300)), whisper_lang_code(language))
    else:
        chunks = [text]
    if not chunks:
        chunks = [text]

    clone_prompt = None
    if mode == "clone":
        clone_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)

    pieces = []
    for index, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"🎙️  Generating chunk {index + 1}/{len(chunks)} ({len(chunk)} chars) ...")

        kw = dict(base_kw, text=chunk)
        if clone_prompt is not None:
            kw["voice_clone_prompt"] = clone_prompt
        elif mode == "design" and instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            where = f" on chunk {index + 1}/{len(chunks)}" if len(chunks) > 1 else ""
            return None, f"Error{where}: {type(e).__name__}: {e}"

        pieces.append(np.asarray(audio[0], dtype=np.float32).reshape(-1))

        # Pin every later chunk to the voice the first one happened to produce.
        if mode == "design" and clone_prompt is None and len(chunks) > 1:
            clone_prompt = _voice_lock_prompt(pieces[0], chunks[0])

    if len(pieces) == 1:
        full = pieces[0]
    else:
        gap = np.zeros(int(sampling_rate * max(0.0, float(chunk_pause or 0.0))), dtype=np.float32)
        joined = []
        for index, piece in enumerate(pieces):
            if index:
                joined.append(gap)
            joined.append(piece)
        full = np.concatenate(joined)

    # Samples can sit a hair above 1.0; without the clip they wrap around to
    # full-scale negative and turn into loud clicks.
    waveform = (np.clip(full, -1.0, 1.0) * 32767).astype(np.int16)

    status = f"Done. Seed {used_seed}"
    if len(chunks) > 1:
        status += f" · joined {len(chunks)} chunks"
    return (sampling_rate, waveform), status

# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.1em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}

/* CSS for Event Tags */
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

def _lang_dropdown(label="Language (optional)", value="English"):
    return gr.Dropdown(
        label=label, choices=_ALL_LANGUAGES, value=value,
        allow_custom_value=False, interactive=True,
    )

def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed", info="1.0 = normal. >1 faster, <1 slower.")
        du = gr.Number(value=None, label="Duration (seconds)", info="Set a fixed duration to override speed.")
        ns = gr.Slider(4, 64, value=32, step=1, label="Inference Steps", info="Lower = faster, higher = better quality.")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
        pp = gr.Checkbox(label="Preprocess Prompt", value=True, info="Applies silence removal and trims reference audio.")
        po = gr.Checkbox(label="Postprocess Output", value=True, info="Removes long silences from generated audio.")
        sd = gr.Number(
            value=-1, precision=0, label="Seed",
            info="-1 picks a new voice every run. Copy the seed from Status to repeat a take exactly."
        )

        with gr.Accordion("Long text", open=False):
            sl = gr.Checkbox(
                label="Auto-split long text", value=True,
                info="Splits on sentence boundaries and joins the audio. Ignored when Duration is set."
            )
            cc = gr.Slider(80, 600, value=300, step=10, label="Max characters per chunk")
            cp = gr.Slider(0.0, 1.5, value=0.35, step=0.05, label="Pause between chunks (seconds)")

    return ns, gs, dn, sp, du, pp, po, sd, sl, cc, cp

with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ OmniVoice Multilingual </h1>
            <p>State-of-the-art text-to-speech model for 600+ languages, supporting Voice Clone and Voice Design.</p>
        </div>
    """)

    with gr.Tabs():
        # ==============================================================
        # Voice Clone Tab
        # ==============================================================
        with gr.TabItem("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Added elem_id for JS hook
                    vc_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter the text to synthesize...", elem_id="vc_textbox")

                    # Tag Buttons for Voice Clone
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vc_text],
                                outputs=vc_text,
                                js=INSERT_TAG_JS_VC
                            )

                    with gr.Row():
                      vc_lang = _lang_dropdown("Language (optional)")
                      vc_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vc_ref_audio = gr.Audio(label="Reference Audio (3–10 seconds audio)", type="filepath", elem_classes="compact-audio")

                    vc_ref_text = gr.Textbox(
                        label="Reference Text", lines=2,
                        placeholder="Auto-transcribed upon audio upload. You can manually edit it if Whisper gets it wrong."
                    )

                    vc_btn = gr.Button("Generate", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po, vc_sd, vc_sl, vc_cc, vc_cp = _gen_settings()

                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="Output Audio", type="numpy")
                    vc_status = gr.Textbox(label="Status", lines=1)

                    with gr.Accordion("Download files", open=False):
                        vc_out_wav = gr.File(label="Generated Audio (WAV)")
                        vc_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vc_out_word_srt = gr.File(label="Word Level SRT")
                        vc_out_shorts_srt = gr.File(label="Shorts SRT")

            def _auto_transcribe(audio_path, lang):
                if not audio_path:
                    return gr.update(value="")
                try:
                    whisper_results = subtitle_maker(audio_path, whisper_lang_code(lang))
                    if whisper_results and len(whisper_results) > 7:
                        return gr.update(value=whisper_results[7])
                except Exception as e:
                    logging.warning(f"Auto-transcription failed: {e}")
                return gr.update(value="")

            vc_ref_audio.change(
                fn=_auto_transcribe,
                inputs=[vc_ref_audio, vc_lang],
                outputs=[vc_ref_text]
            )

            def _clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po,
                          sd, sl, cc, cp):
                res = _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone",
                                ref_text=ref_text, seed=sd, auto_split=sl, chunk_chars=cc, chunk_pause=cp)
                if res[0] is None:
                    return None, res[1], None, None, None, None

                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)

                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)

                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vc_btn.click(
                _clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po,
                        vc_sd, vc_sl, vc_cc, vc_cp],
                outputs=[vc_audio, vc_status, vc_out_wav, vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt],
            )

        # ==============================================================
        # Voice Design Tab
        # ==============================================================
        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Added elem_id for JS hook
                    vd_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter the text to synthesize...", elem_id="vd_textbox")

                    # Tag Buttons for Voice Design
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vd_text],
                                outputs=vd_text,
                                js=INSERT_TAG_JS_VD
                            )

                    with gr.Row():
                      vd_lang = _lang_dropdown(value='English')
                      vd_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vd_btn = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Character Voice Design", open=False):
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            default_val = "Auto"
                            if _cat == "Gender":
                                default_val = "Female"
                            elif _cat == "Age":
                                default_val = "Young Adult"

                            vd_groups.append(
                                gr.Dropdown(label=_cat, choices=["Auto"] + _choices, value=default_val, info=_ATTR_INFO.get(_cat))
                            )

                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po, vd_sd, vd_sl, vd_cc, vd_cp = _gen_settings()

                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="Output Audio", type="numpy")
                    vd_status = gr.Textbox(label="Status", lines=1)

                    with gr.Accordion("Download files", open=False):
                        vd_out_wav = gr.File(label="Generated Audio (WAV)")
                        vd_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vd_out_word_srt = gr.File(label="Word Level SRT")
                        vd_out_shorts_srt = gr.File(label="Shorts SRT")

            def _build_instruct(groups):
                selected = [g for g in groups if g and g != "Auto"]
                if not selected: return None
                return ", ".join([DIALECT_MAP.get(v, v) for v in selected])

            def _design_fn(text, lang, want_subs, ns, gs, dn, sp, du, pp, po, sd, sl, cc, cp, *groups):
                instruct = _build_instruct(groups)
                res = _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design",
                                seed=sd, auto_split=sl, chunk_chars=cc, chunk_pause=cp)
                if res[0] is None:
                    return None, res[1], None, None, None, None

                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)

                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)

                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vd_btn.click(
                _design_fn,
                inputs=[vd_text, vd_lang, vd_want_subs, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po,
                        vd_sd, vd_sl, vd_cc, vd_cp] + vd_groups,
                outputs=[vd_audio, vd_status, vd_out_wav, vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt],
            )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
