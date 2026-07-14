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
4. In the **Run Gradio APP** cell, paste your Hugging Face token into the `HF_TOKEN` field (free — create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), type *Read*). Without a token, Hugging Face heavily throttles the 3.3 GB model download.
5. Run the cell and open the public `*.gradio.live` link.

> **Model download stuck / very slow?** Make sure `HF_TOKEN` is set. If huggingface.co itself is slow or blocked in your region, also tick `Use_HF_Mirror`.

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
