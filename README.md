# Colab Notebooks

Two one-click Google Colab apps. Press a badge, run the cells top to bottom, open the
`*.gradio.live` link that appears.

| Notebook | What it does | Open |
|---|---|---|
| **FLUX.1 Image Generator** | Text → image with FLUX.1-dev (A100 friendly) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hirannalaka19/omnivoice-colab/blob/main/FLUX_Colab.ipynb) |
| **OmniVoice** | Text → speech, voice cloning, subtitles | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hirannalaka19/omnivoice-colab/blob/main/OmniVoice_Colab.ipynb) |

---
---

# 🎨 FLUX.1 Image Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hirannalaka19/omnivoice-colab/blob/main/FLUX_Colab.ipynb)

Run **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)** by Black Forest Labs
on a Colab **A100**, through a Gradio UI — no local GPU, no setup beyond a Hugging Face token.

## 🔹 What it can do

* **Text → image** with FLUX.1-dev, FLUX.1-Krea-dev, FLUX.1-schnell, or any custom repo id
* **Image → image** — redraw an upload, with a strength slider
* **LoRA** — load any FLUX LoRA from the Hub on top of the base model, with a scale slider
* 8 aspect-ratio presets plus free width/height (snapped to multiples of 16)
* Seed control for reproducible images, batches of up to 8, live per-step progress
* Every PNG is saved with its prompt, seed and settings in the metadata, plus a ZIP download
* Picks its own precision from the GPU it gets — full bf16, CPU offload, or 4-bit NF4

## 💻 Which GPU

Set it under `Runtime → Change runtime type`.

| GPU | Precision used | ~1024×1024, 28 steps |
|---|---|---|
| **A100 40GB** (Colab Pro) | bf16, whole pipeline on GPU | ~10–15 s |
| **L4 24GB** | bf16 + CPU offload | ~50–70 s |
| **T4 16GB** (free tier) | 4-bit NF4 | ~2–4 min |

The first run of a session also downloads ~34 GB of weights — that is one-off, not per image.

## 🔑 One-time setup

FLUX.1-dev is a **gated** model, so two things are needed before it will download:

1. Open <https://huggingface.co/black-forest-labs/FLUX.1-dev> while signed in to Hugging Face
   and click **Agree and access repository**.
2. Create a **Read** token at <https://huggingface.co/settings/tokens>, then in Colab click the
   **🔑 key icon** in the left sidebar → *Add new secret* → Name `HF_TOKEN`, paste the token,
   switch on **Notebook access**. It persists across every future session.

> Don't want to do either? Pick `black-forest-labs/FLUX.1-schnell` in step 3 of the notebook —
> it is ungated, Apache-2.0, and only needs 4 steps per image.

## 🚀 How to use

1. Click the **Open In Colab** badge above.
2. `Runtime → Change runtime type → A100 GPU → Save`.
3. Run **1. Check the GPU**.
4. Run **2. Install FLUX + the app** (~2 min).
5. Run **3. Download the model weights** (~34 GB, a few minutes).
6. Run **4. Run the FLUX image generator**, then open the `https://….gradio.live` link.
   Leave that cell running — stopping it closes the app.

Images land in `/content/Flux_Output`. Colab deletes that when the runtime disconnects, so use
the optional **Copy to Google Drive** cell if you want to keep them.

## ✍️ Prompting notes

FLUX.1-dev is guidance-distilled, so there is **no negative prompt** — describe what you *do*
want, in ordinary sentences. It handles long prompts well, is unusually good at rendering
**text inside images** (put the words in quotes), and looks most natural at guidance 2.5–4.

## 🧰 Run it locally

```bash
git clone https://github.com/hirannalaka19/omnivoice-colab.git
cd omnivoice-colab
pip install torch --index-url https://download.pytorch.org/whl/cu124   # if you don't have it
pip install -r flux_colab.txt
export HF_TOKEN=hf_your_token_here
python flux_app.py
```

## ⚖️ Licence

FLUX.1-dev and FLUX.1-Krea-dev are under the
[FLUX.1 \[dev\] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
— personal, research and evaluation use only, **not commercial**. FLUX.1-schnell is Apache-2.0
and may be used commercially. You are responsible for what you generate; follow the Black Forest
Labs Acceptable Use Policy.

---
---

# 🎙️ Run OmniVoice On Google Colab

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
* Narrate long scripts — text is split on sentence boundaries and stitched back together
* Reproducible takes via a seed

---

## 🎚️ Long text & seeds

Both are under **Generation Settings**.

**Seed** — generation is random, so the same text gives a slightly different voice
every run. Leave it at `-1` to get a new voice each time; the seed that was used is
printed in the **Status** box. Paste that number back into Seed to reproduce that exact
take.

**Auto-split long text** (on by default) — anything longer than the chunk size is split
on sentence boundaries, generated piece by piece, and joined with a short pause. Event
tags like `[laughter]` are never split. In Voice Design the voice from the first chunk is
reused for the rest, so a long script stays in one voice.

Setting a fixed **Duration** turns splitting off, since a duration describes a single
utterance.

---

## 🚀 How to use

1. Click the **Open In Colab** badge above.
2. Set the runtime to **GPU** (T4 is enough): `Runtime → Change runtime type → T4 GPU`.
3. **One-time:** add your Hugging Face token to Colab **Secrets** — click the 🔑 icon in the left sidebar → *Add new secret* → Name: `HF_TOKEN`, Value: your token (create one free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), type *Read*) → enable *Notebook access*. Hugging Face throttles/blocks anonymous downloads, so the token is required for full download speed.
4. Run the **Install OmniVoice** cell.
5. Run the **Run Gradio APP** cell and open the public `*.gradio.live` link.

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

* 👨‍💻 Colab wrappers & Gradio apps by [HiranNalaka](https://github.com/hirannalaka19)
* 👉 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — the original OmniVoice model
* 👉 [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) — the FLUX.1 model
* 👉 [huggingface/diffusers](https://github.com/huggingface/diffusers) — FLUX inference

---

## ⚠️ Disclaimer

Please use these models responsibly. Do not use them for harmful, misleading, or unethical
purposes such as unauthorized voice cloning, impersonation, fraud, scams, deceptive deepfakes,
or any illegal content. You are responsible for what you generate.
