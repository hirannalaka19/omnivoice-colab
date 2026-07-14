# Run OmniVoice On Google Colab

Run **OmniVoice** easily on Google Colab, no complex setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hirannalaka19/omnivoice-colab/blob/main/OmniVoice_Colab.ipynb)

---

## 🧠 About

This is a **Google Colab version** of the original [OmniVoice](https://github.com/k2-fsa/OmniVoice) model.
It allows you to quickly generate high-quality speech from text with minimal setup.

Uses the **official `omnivoice` pip package (v0.2.0+)** — no source fork needed.

---

## 🔹 What it can do

* Convert text → speech
* Clone voices from audio (zero-shot)
* Support 600+ languages ([language list](https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md))
* Add emotions using tags (`[laughter]`, `[sigh]`, etc.)
* Voice Design — customize voice (gender, age, pitch, accent, style)
* Fast inference (RTF as low as 0.025)
* Generate subtitles (SRT) — sentence level, word level, and Shorts style

---

## 🚀 How to use

1. Click the **Open In Colab** badge above.
2. Set the runtime to **GPU** (T4 is enough): `Runtime → Change runtime type → T4 GPU`.
3. Run the **Install OmniVoice** cell.
4. In the **Run Gradio APP** cell, paste your Hugging Face token into the `HF_TOKEN` field (free — create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), type *Read*). Hugging Face throttles/blocks anonymous downloads, so a token gives full download speed.
5. Run the cell and open the public `*.gradio.live` link.

> **Note on model download:** the cell tests huggingface.co download speed first — if it's good, the model downloads from Hugging Face; if it's slow or blocked, it automatically switches to hf-mirror.com. As a last resort the app falls back to the official [ModelScope mirror](https://www.modelscope.cn/models/k2-fsa/OmniVoice).

---

## 💻 Run locally

```bash
git clone https://github.com/hirannalaka19/omnivoice-colab.git
cd omnivoice-colab
pip install -r requirements.txt
python app.py
```

---

## 🙌 Credit

* 👨‍💻 Colab wrapper & Gradio app by [HiranNalaka](https://github.com/hirannalaka19)
* 👉 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — the original OmniVoice model

---

## ⚠️ Disclaimer

Please use this model responsibly. Do not use it for harmful, misleading, or unethical purposes such as unauthorized voice cloning, impersonation, fraud, or scams.
