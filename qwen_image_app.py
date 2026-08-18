"""Qwen-Image generator - Gradio app, tuned for Google Colab (A100 friendly).

Wraps Qwen/Qwen-Image - a 20B MMDiT text-to-image model from Alibaba's Qwen team,
Apache-2.0 - in a small Gradio UI, so a Colab notebook becomes a text-to-image
studio with a public link.

Unlike FLUX.1-dev, Qwen-Image is **ungated**: no licence to accept, no token
required. An HF_TOKEN still helps, because anonymous Hub downloads are throttled.

The model is big - 57 GB of bf16 weights, of which the transformer alone is
41 GB - so the app picks a loading strategy from the GPU it was actually handed,
and can pull the Lightning LoRAs to cut 50 denoising steps down to 8 or 4.
"""

import gc
import json
import math
import os
import random
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, PngImagePlugin

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# On Colab everything lives under /content; locally keep it next to the script.
OUTPUT_DIR = (Path("/content/Qwen_Output") if Path("/content").is_dir() else Path("Qwen_Output")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEED = 2**31 - 1

MODELS = {
    "Qwen-Image  -  20B, best quality (57 GB download)": {
        "repo": "Qwen/Qwen-Image",
    },
    "Qwen-Image-2512  -  newer December 2025 release (57 GB)": {
        "repo": "Qwen/Qwen-Image-2512",
    },
    "Qwen-Image NF4  -  same model pre-quantised, half the download (28 GB)": {
        "repo": "diffusers/qwen-image-nf4",
    },
    "Custom repo id...": {
        "repo": "",
    },
}
DEFAULT_MODEL = list(MODELS)[0]
CUSTOM_MODEL = "Custom repo id..."

# Lightning distils the 50-step schedule down to 8 or 4 steps. Keyed by base repo,
# because Qwen-Image and Qwen-Image-2512 need LoRAs trained on their own weights.
# The bf16 files are half the size of the fp32 ones and identical for bf16 inference.
QWEN_IMAGE_LIGHTNING = (
    "lightx2v/Qwen-Image-Lightning",
    {
        8: "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
        4: "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
    },
)
LIGHTNING = {
    "Qwen/Qwen-Image": QWEN_IMAGE_LIGHTNING,
    "diffusers/qwen-image-nf4": QWEN_IMAGE_LIGHTNING,
    "Qwen/Qwen-Image-2512": (
        "lightx2v/Qwen-Image-2512-Lightning",
        {
            8: "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors",
            4: "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        },
    ),
}

# Lightning was distilled with shift=3 and no terminal shift, so it needs its own
# scheduler settings - the stock ones give mush at 4-8 steps. Verbatim from
# ModelTC/Qwen-Image-Lightning/generate_with_diffusers.py.
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

SPEED = {
    "Quality  -  50 steps, the base model": None,
    "Fast  -  8 steps, Lightning LoRA": 8,
    "Turbo  -  4 steps, Lightning LoRA": 4,
}
DEFAULT_SPEED = list(SPEED)[1]  # Lightning is the sane default on a Colab GPU.

# The resolutions Qwen-Image was tuned on, from the official reference script.
# All are multiples of 16: the VAE downscales 8x, then the transformer patches 2x2.
ASPECTS = {
    "1:1   square       1328 x 1328": (1328, 1328),
    "16:9  wide         1664 x 928": (1664, 928),
    "9:16  phone         928 x 1664": (928, 1664),
    "4:3   landscape    1472 x 1104": (1472, 1104),
    "3:4   portrait     1104 x 1472": (1104, 1472),
    "3:2   landscape    1584 x 1056": (1584, 1056),
    "2:3   portrait     1056 x 1584": (1056, 1584),
    "1:1   quick        1024 x 1024": (1024, 1024),
    "Custom (use the sliders below)": None,
}
DEFAULT_ASPECT = list(ASPECTS)[0]

PRECISION = {
    "Auto  -  pick from the detected GPU": "auto",
    "bf16 full GPU  -  fastest, needs ~62 GB (A100 80GB, H100)": "bf16",
    "bf16 + CPU offload  -  needs ~40 GB VRAM and ~64 GB RAM": "offload",
    "bf16 + group offload  -  experimental: full quality on a 24-40 GB card, slow": "group",
    "4-bit NF4  -  ~17 GB VRAM (A100 40GB, L4, T4)": "nf4",
}
DEFAULT_PRECISION = list(PRECISION)[0]

# Qwen's own "make it look good" suffix, appended when the checkbox is on.
POSITIVE_MAGIC = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图.",
}

# Qwen-Image is a true CFG model, so unlike FLUX a negative prompt does something.
# A single space is what the reference script uses when you have nothing to say.
DEFAULT_NEGATIVE = " "

EXAMPLE_PROMPTS = [
    'a bookshop window at dusk, a hand-lettered chalkboard sign reading "OPEN LATE - POETRY UPSTAIRS", warm lamplight, rain on the glass, 35mm photograph',
    "a golden retriever puppy asleep on a stack of old books, warm window light, 85mm photograph, shallow depth of field",
    'a movie poster titled "IMAGINATION UNLEASHED", subtitle "Enter a world beyond your imagination", a sleek futuristic computer erupting with swirling colour and whimsical creatures, cinematic sci-fi surrealism, ultra detailed',
    "an isometric miniature diorama of a cozy ramen shop at night, tilt-shift, soft neon signage, highly detailed",
    "一副典雅庄重的对联悬挂于厅堂之中，房间是安静古典的中式布置，桌子上放着青花瓷，对联左书“义本生知人机同道善思新”，右书“通云赋智乾坤启数高志远”，横批“智启通义”，字体飘逸",
    "studio product shot of a matte black ceramic coffee mug on polished concrete, softbox lighting, minimal, editorial",
    'an infographic poster explaining the water cycle, clean flat vector style, labelled arrows reading "EVAPORATION", "CONDENSATION", "PRECIPITATION", muted palette',
    "portrait of an elderly fisherman in Galle, weathered face, rim light, shot on Kodak Portra 400, natural skin texture",
]

CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ]")


# --------------------------------------------------------------------------
# Hardware helpers
# --------------------------------------------------------------------------

def gpu_name_and_vram():
    """Return (name, total VRAM in GiB). VRAM is 0 when there is no CUDA GPU."""
    if not torch.cuda.is_available():
        return "no CUDA GPU", 0.0
    props = torch.cuda.get_device_properties(0)
    return props.name, props.total_memory / 1024**3


def vram_used():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(0) / 1024**3


def auto_precision():
    """Pick a loading strategy that fits the GPU we were actually handed.

    The bf16 transformer alone is 38 GiB, so a 40 GB A100 cannot hold it plus
    activations - that card gets NF4, not full precision. Only 80 GB cards run
    the whole pipeline resident.
    """
    _, total = gpu_name_and_vram()
    if total >= 62:      # A100 80GB, H100 - everything stays on the GPU
        return "bf16"
    if total >= 42:      # 48 GB cards - swap whole modules in and out
        return "offload"
    return "nf4"         # A100 40GB, L4, T4 - quantise the two big blocks


def hardware_banner():
    name, total = gpu_name_and_vram()
    if total == 0:
        return (
            "### No GPU detected\n"
            "In Colab: **Runtime -> Change runtime type -> GPU -> A100** "
            "(A100 needs Colab Pro; L4 and T4 also work, just slower)."
        )
    picked = [k for k, v in PRECISION.items() if v == auto_precision()][0]
    tier = "plenty of headroom" if total >= 62 else ("comfortable" if total >= 38 else "tight, expect slow runs")
    return (
        f"### GPU: **{name}** - {total:.0f} GB VRAM ({tier})\n"
        f"Auto precision will use: **{picked.split('  -  ')[0]}**"
    )


def hf_token():
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def lightning_for(repo):
    """(lora repo, {steps: filename}) for a base model. Custom repos get the plain
    Qwen-Image LoRAs, which is right for most Qwen-Image derivatives."""
    return LIGHTNING.get(repo, QWEN_IMAGE_LIGHTNING)


# --------------------------------------------------------------------------
# peft / torchao workaround
# --------------------------------------------------------------------------

_PEFT_TORCHAO_CHECKED = False


def allow_lora_without_torchao():
    """Let LoRAs load when the installed torchao is older than peft would like.

    While picking a layer class for each Linear it patches, peft asks whether torchao
    is available. If torchao is installed but too old, that probe *raises* instead of
    returning False, and the whole LoRA load dies with it - even though nothing here
    is torchao-quantised, since this app quantises with bitsandbytes. Colab currently
    ships torchao 0.10 against a peft that wants 0.16+, which is exactly that trap.

    Nothing is installed or removed: peft is simply told torchao is not there, which
    is the answer it would have given if Colab had not shipped torchao at all. Only
    done when the probe actually raises, so a healthy environment is left alone.
    """
    global _PEFT_TORCHAO_CHECKED
    if _PEFT_TORCHAO_CHECKED:
        return
    _PEFT_TORCHAO_CHECKED = True

    import sys

    try:
        from peft import import_utils
    except Exception:
        return

    try:
        import_utils.is_torchao_available()
        return  # answered with a bool, so peft is happy either way
    except ImportError as err:
        print(f"[qwen] peft's torchao check would break LoRA loading, disabling it ({err})")
    except Exception:
        return

    def torchao_is_not_here():
        return False

    # peft's dispatchers did `from peft.import_utils import is_torchao_available`, so
    # the name lives in their namespaces too and has to be rebound in each of them.
    try:
        import peft.tuners.lora.torchao  # noqa: F401 - ensure it exists before patching
    except Exception:
        pass

    import_utils.is_torchao_available = torchao_is_not_here
    for name, module in list(sys.modules.items()):
        if name.startswith("peft") and getattr(module, "is_torchao_available", None) is not None:
            module.is_torchao_available = torchao_is_not_here


# --------------------------------------------------------------------------
# Pipeline manager
# --------------------------------------------------------------------------

# Answers to "is this repo already quantised", so a HEAD request is not made on
# every single generate call.
_PREQUANT_CACHE = {}


class QwenRunner:
    """Holds one loaded Qwen-Image pipeline and reuses it across generations."""

    def __init__(self):
        self.pipe = None
        self.i2i = None
        self.repo = None
        self.mode = None
        self.base_scheduler = None   # the repo's own scheduler config
        self.speed = None            # None, or the Lightning step count in use
        self.lora = None             # user LoRA source, on top of Lightning
        self.lora_scale = 1.0

    # -- memory ------------------------------------------------------------

    def unload(self):
        self.pipe = None
        self.i2i = None
        self.repo = None
        self.mode = None
        self.base_scheduler = None
        self.speed = None
        self.lora = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # -- loading -----------------------------------------------------------

    @staticmethod
    def _is_prequantized(repo, token):
        """True when the repo already ships bitsandbytes-quantised weights - those
        cannot be quantised again, and must not be moved with .to()."""
        if repo in _PREQUANT_CACHE:
            return _PREQUANT_CACHE[repo]
        answer = False
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo, "transformer/config.json", token=token)
            with open(path, encoding="utf-8") as handle:
                answer = "quantization_config" in json.load(handle)
        except Exception:
            answer = False
        _PREQUANT_CACHE[repo] = answer
        return answer

    def _load_nf4(self, repo, token):
        """4-bit the two heavyweights (20B transformer, 7B Qwen2.5-VL encoder) so a
        40 GB A100 - or a 16 GB T4 - can hold the whole thing."""
        from diffusers import BitsAndBytesConfig as DiffusersBnb
        from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

        transformer = QwenImageTransformer2DModel.from_pretrained(
            repo,
            subfolder="transformer",
            quantization_config=DiffusersBnb(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
            token=token,
        )

        text_encoder = None
        try:
            from transformers import BitsAndBytesConfig as TransformersBnb
            from transformers import Qwen2_5_VLForConditionalGeneration

            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                repo,
                subfolder="text_encoder",
                quantization_config=TransformersBnb(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                torch_dtype=torch.bfloat16,
                token=token,
            )
        except Exception as err:  # noqa: BLE001 - transformers may have moved the class
            print(
                f"[qwen] text encoder left in bf16 ({type(err).__name__}: {err}) "
                "- it needs ~17 GB of VRAM while encoding the prompt."
            )

        kwargs = {"transformer": transformer, "torch_dtype": torch.bfloat16, "token": token}
        if text_encoder is not None:
            kwargs["text_encoder"] = text_encoder
        pipe = QwenImagePipeline.from_pretrained(repo, **kwargs)
        # bitsandbytes modules cannot be moved with .to(), so offload hooks are the way.
        pipe.enable_model_cpu_offload()
        return pipe

    def _load_group(self, repo, token):
        """Full bf16 quality on a small card: move the transformer's leaves on and off
        the GPU as they are needed. Needs ~60 GB of system RAM, so a high-RAM runtime."""
        from diffusers import QwenImagePipeline
        from diffusers.hooks import apply_group_offloading

        pipe = QwenImagePipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, token=token)
        onload, offload = torch.device("cuda"), torch.device("cpu")
        # use_stream would prefetch the next block, but it pins ~57 GB of host memory
        # and Colab tends to die doing that. Trade the speed for surviving.
        pipe.transformer.enable_group_offload(
            onload_device=onload, offload_device=offload, offload_type="leaf_level", use_stream=False
        )
        apply_group_offloading(
            pipe.text_encoder,
            onload_device=onload,
            offload_device=offload,
            offload_type="block_level",
            num_blocks_per_group=2,
        )
        pipe.vae.to(onload)
        return pipe

    def load(self, repo, mode, progress=None):
        if not repo:
            raise gr.Error("Pick a model, or type a Hugging Face repo id in the custom field.")
        if not torch.cuda.is_available():
            raise gr.Error("No GPU. In Colab: Runtime -> Change runtime type -> GPU (A100 recommended).")

        token = hf_token()
        if mode == "auto":
            mode = auto_precision()
        if self._is_prequantized(repo, token):
            mode = "prequantized"
        if self.pipe is not None and self.repo == repo and self.mode == mode:
            return mode

        self.unload()
        if progress:
            size = "28 GB" if mode == "prequantized" else "57 GB"
            progress(0.05, desc=f"Loading {repo} - the first run downloads ~{size}, this takes a while...")

        try:
            if mode == "nf4":
                pipe = self._load_nf4(repo, token)
            elif mode == "group":
                pipe = self._load_group(repo, token)
            else:
                from diffusers import QwenImagePipeline

                pipe = QwenImagePipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, token=token)
                if mode == "bf16":
                    pipe.to("cuda")
                else:
                    # 'offload' and 'prequantized' both ride on module-level offload
                    # hooks: one submodule on the GPU at a time.
                    pipe.enable_model_cpu_offload()
        except Exception as err:  # noqa: BLE001 - re-raised as a readable gr.Error
            text = f"{type(err).__name__} {err}".lower()
            if any(s in text for s in ("gated", "401", "403", "restricted", "unauthorized")):
                raise gr.Error(
                    f"The Hub refused to serve '{repo}'. Qwen-Image itself is ungated, so this is "
                    "usually a private or renamed repo, or a rate limit on anonymous downloads - "
                    "add an HF_TOKEN Colab secret and re-run the cell.\n\n"
                    f"(underlying error: {type(err).__name__}: {str(err)[:300]})"
                )
            if "out of memory" in text:
                raise gr.Error(
                    "Ran out of memory while loading. Set Precision to '4-bit NF4', or pick the "
                    "'Qwen-Image NF4' model, and try again."
                )
            if "qwenimage" in text and ("cannot import" in text or "no attribute" in text):
                raise gr.Error(
                    "This diffusers build has no Qwen-Image support. Re-run the install cell, or "
                    "pip install -U 'diffusers>=0.36.0'."
                )
            raise gr.Error(f"Could not load '{repo}': {type(err).__name__}: {str(err)[:400]}")

        pipe.set_progress_bar_config(disable=True)
        # Slicing is free. Tiling is only worth it when memory is tight: it can leave
        # faint seams, which an 80 GB card has no reason to risk.
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        if mode in ("nf4", "group", "prequantized"):
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass

        self.pipe = pipe
        self.repo = repo
        self.mode = mode
        self.i2i = None
        self.lora = None
        self.speed = None
        self.base_scheduler = {k: v for k, v in dict(pipe.scheduler.config).items() if not k.startswith("_")}
        return mode

    def img2img(self):
        """Image-to-image pipeline sharing the loaded weights (costs no extra VRAM)."""
        if self.pipe is None:
            raise gr.Error("Load a model first - generate one image on the 'Text to Image' tab.")
        if self.i2i is None:
            from diffusers import QwenImageImg2ImgPipeline

            # from_pipe reuses the same modules and keeps any offload hooks.
            self.i2i = QwenImageImg2ImgPipeline.from_pipe(self.pipe)
            self.i2i.set_progress_bar_config(disable=True)
        return self.i2i

    # -- Lightning ---------------------------------------------------------

    def set_speed(self, want, progress=None):
        """want is None for the base 50-step model, or 8 / 4 for a Lightning LoRA."""
        if self.pipe is None or self.speed == want:
            return
        from diffusers import FlowMatchEulerDiscreteScheduler

        if self.speed is not None:
            try:
                self.pipe.delete_adapters("lightning")
            except Exception:
                pass

        if want is None:
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.base_scheduler)
        else:
            lora_repo, files = lightning_for(self.repo)
            weight_name = files.get(want)
            if progress:
                progress(0.08, desc=f"Fetching the {want}-step Lightning LoRA (~0.9 GB, once)...")
            allow_lora_without_torchao()
            try:
                self.pipe.load_lora_weights(
                    lora_repo, weight_name=weight_name, adapter_name="lightning", token=hf_token()
                )
            except Exception as err:  # noqa: BLE001
                # A failed injection can leave half-patched layers behind, so clear the
                # adapter before handing the error on - a retry then starts clean.
                try:
                    self.pipe.delete_adapters("lightning")
                except Exception:
                    pass
                raise gr.Error(
                    f"Could not load the Lightning LoRA ({weight_name}): "
                    f"{type(err).__name__}: {str(err)[:300]}\n\n"
                    "Switch Speed back to 'Quality' to use the base model instead."
                )
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)

        self.speed = want
        self._sync_adapters()
        self.i2i = None  # rebuild so the img2img pipe sees the change

    # -- user LoRA ---------------------------------------------------------

    def _sync_adapters(self):
        """Lightning and a user LoRA can stack, so set the whole list at once."""
        names, weights = [], []
        if self.speed is not None:
            names.append("lightning")
            weights.append(1.0)
        if self.lora:
            names.append("user_lora")
            weights.append(float(self.lora_scale))
        if not names:
            return
        try:
            self.pipe.set_adapters(names, adapter_weights=weights)
        except Exception as err:  # noqa: BLE001
            print(f"[qwen] could not set adapters {names}: {err}")

    def apply_lora(self, source, weight_name, scale):
        if self.pipe is None:
            raise gr.Error("Load a model first (generate one image), then apply a LoRA.")
        source = (source or "").strip()
        if not source:
            raise gr.Error("Enter a LoRA repo id (e.g. flymy-ai/qwen-image-realism-lora) or a .safetensors path.")
        self.remove_lora()
        kwargs = {"adapter_name": "user_lora"}
        if (weight_name or "").strip():
            kwargs["weight_name"] = weight_name.strip()
        token = hf_token()
        if token:
            kwargs["token"] = token
        allow_lora_without_torchao()
        try:
            self.pipe.load_lora_weights(source, **kwargs)
        except Exception as err:  # noqa: BLE001
            try:
                self.pipe.delete_adapters("user_lora")
            except Exception:
                pass
            raise gr.Error(f"LoRA failed to load: {type(err).__name__}: {str(err)[:300]}")
        self.lora = source
        self.lora_scale = float(scale)
        self._sync_adapters()
        self.i2i = None
        stacked = " (stacked on the Lightning LoRA)" if self.speed else ""
        return f"LoRA active: `{source}` at scale {float(scale):.2f}{stacked}"

    def remove_lora(self):
        if self.pipe is None:
            return "Nothing loaded yet."
        try:
            self.pipe.delete_adapters("user_lora")
        except Exception:
            pass
        self.lora = None
        self.i2i = None
        self._sync_adapters()
        return "LoRA removed - back to the base model."


RUNNER = QwenRunner()


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

def save_image(image, prompt, negative, seed, steps, cfg, repo, size, speed):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"qwen_{stamp}_{seed}.png"
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "parameters",
        f"{prompt}\n"
        f"Negative prompt: {negative}\n"
        f"Model: {repo}, Steps: {steps}, True CFG scale: {cfg}, Seed: {seed}, "
        f"Size: {size[0]}x{size[1]}, Lightning: {speed or 'off'}",
    )
    image.save(path, pnginfo=info)
    return path


def zip_paths(paths):
    if not paths:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = OUTPUT_DIR / f"qwen_batch_{stamp}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, Path(p).name)
    return str(archive)


def resolve_model(choice, custom_repo):
    spec = MODELS.get(choice, MODELS[DEFAULT_MODEL])
    return spec["repo"] or (custom_repo or "").strip(), spec


def snap16(value):
    """VAE 8x downscale plus 2x2 patching means both sides must divide by 16."""
    return max(256, int(round(float(value) / 16) * 16))


def decorate(prompt, magic):
    """Qwen ships a quality suffix, in two languages. Pick by what the prompt is in."""
    if not magic:
        return prompt
    return prompt + POSITIVE_MAGIC["zh" if CJK.search(prompt) else "en"]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate(
    prompt,
    negative,
    model_choice,
    custom_repo,
    precision_choice,
    speed_choice,
    aspect,
    width,
    height,
    steps,
    cfg,
    seed,
    randomize,
    count,
    magic,
    max_seq,
    progress=gr.Progress(),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write a prompt first.")

    repo, _spec = resolve_model(model_choice, custom_repo)
    mode = PRECISION.get(precision_choice, "auto")
    want_speed = SPEED.get(speed_choice)

    progress(0.02, desc="Preparing model...")
    t_load = time.time()
    used_mode = RUNNER.load(repo, mode, progress=progress)
    RUNNER.set_speed(want_speed, progress=progress)
    load_secs = time.time() - t_load

    if ASPECTS.get(aspect):
        width, height = ASPECTS[aspect]
    width, height = snap16(width), snap16(height)

    steps = int(steps)
    count = max(1, int(count))
    cfg = float(cfg)
    # Lightning is distilled without classifier-free guidance: a true_cfg_scale above
    # 1 both doubles the work and wrecks the result.
    if want_speed is not None:
        cfg = 1.0
    full_prompt = decorate(prompt, magic)
    negative = negative if (negative or "").strip() else DEFAULT_NEGATIVE

    images, paths, seeds = [], [], []
    base_seed = random.randint(0, MAX_SEED) if randomize else int(seed)
    total_steps = count * steps

    for index in range(count):
        this_seed = (base_seed + index) % MAX_SEED
        # A CPU generator gives identical latents whether or not offload hooks are active.
        generator = torch.Generator("cpu").manual_seed(this_seed)
        done_before = index * steps

        def step_cb(_pipe, i, _t, kwargs, _done=done_before, _n=index):
            progress(
                (_done + i + 1) / total_steps,
                desc=f"Image {_n + 1}/{count} - step {i + 1}/{steps}",
            )
            return kwargs

        call = dict(
            prompt=full_prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            max_sequence_length=int(max_seq),
            generator=generator,
            callback_on_step_end=step_cb,
        )

        started = time.time()
        try:
            result = RUNNER.pipe(**call)
        except TypeError:
            # Older diffusers builds do not accept callback_on_step_end.
            call.pop("callback_on_step_end", None)
            result = RUNNER.pipe(**call)
        except torch.cuda.OutOfMemoryError:
            RUNNER.unload()
            raise gr.Error(
                f"CUDA out of memory at {width}x{height}. Drop to 1024x1024, set the image count "
                "to 1, or set Precision to '4-bit NF4'."
            )
        took = time.time() - started

        image = result.images[0]
        path = save_image(image, full_prompt, negative, this_seed, steps, cfg, repo, (width, height), want_speed)
        images.append(image)
        paths.append(str(path))
        seeds.append(this_seed)

        yield (
            images,
            f"{index + 1}/{count} done - seed `{this_seed}` in {took:.1f}s "
            f"({width}x{height}, {steps} steps)",
            None,
            base_seed,
        )

    archive = zip_paths(paths) if len(paths) > 1 else (paths[0] if paths else None)
    name, total = gpu_name_and_vram()
    status = (
        f"**{len(images)} image(s)** from `{repo}`  \n"
        f"Seeds: `{', '.join(str(s) for s in seeds)}`  \n"
        f"{width}x{height} - {steps} steps - true CFG {cfg} - precision `{used_mode}`"
        + (f" - Lightning {want_speed}-step" if want_speed else "")
        + "  \n"
        + f"GPU {name} ({vram_used():.1f} / {total:.0f} GB in use)"
        + (f"  \nModel load / warm-up: {load_secs:.0f}s" if load_secs > 5 else "")
        + f"  \nSaved to `{OUTPUT_DIR}`"
    )
    yield images, status, archive, base_seed


def generate_img2img(
    prompt,
    negative,
    init_image,
    strength,
    steps,
    cfg,
    seed,
    randomize,
    magic,
    max_seq,
    progress=gr.Progress(),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write a prompt describing what the result should look like.")
    if init_image is None:
        raise gr.Error("Upload a starting image.")
    if RUNNER.pipe is None:
        raise gr.Error("Load a model first - generate one image on the 'Text to Image' tab.")

    pipe = RUNNER.img2img()
    image = init_image.convert("RGB")
    width, height = snap16(image.width), snap16(image.height)
    image = image.resize((width, height), Image.LANCZOS)

    steps = int(steps)
    strength = float(strength)
    cfg = 1.0 if RUNNER.speed is not None else float(cfg)
    negative = negative if (negative or "").strip() else DEFAULT_NEGATIVE
    this_seed = random.randint(0, MAX_SEED) if randomize else int(seed)
    generator = torch.Generator("cpu").manual_seed(this_seed)
    # img2img only walks the tail of the schedule, so that is all the progress bar sees.
    expected = max(1, int(steps * strength))

    def step_cb(_pipe, i, _t, kwargs):
        progress(min(1.0, (i + 1) / expected), desc=f"step {i + 1}/{expected}")
        return kwargs

    full_prompt = decorate(prompt, magic)
    call = dict(
        prompt=full_prompt,
        negative_prompt=negative,
        image=image,
        strength=strength,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        max_sequence_length=int(max_seq),
        generator=generator,
        callback_on_step_end=step_cb,
    )
    started = time.time()
    try:
        result = pipe(**call)
    except TypeError:
        call.pop("callback_on_step_end", None)
        result = pipe(**call)
    except torch.cuda.OutOfMemoryError:
        raise gr.Error("CUDA out of memory - use a smaller input image or 4-bit precision.")

    out = result.images[0]
    path = save_image(
        out, full_prompt, negative, this_seed, steps, cfg, RUNNER.repo or "", (width, height), RUNNER.speed
    )
    status = (
        f"Done in {time.time() - started:.1f}s - seed `{this_seed}`, "
        f"strength {strength:.2f}, {width}x{height}  \nSaved to `{path}`"
    )
    return [out], status, str(path), this_seed


def free_memory():
    RUNNER.unload()
    _, total = gpu_name_and_vram()
    return f"Model unloaded. VRAM in use: {vram_used():.1f} / {total:.0f} GB."


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def on_model_change(choice):
    return gr.update(visible=choice == CUSTOM_MODEL)


def on_speed_change(choice):
    """Lightning fixes both the step count and the guidance, so reflect that."""
    want = SPEED.get(choice)
    if want is None:
        return gr.update(value=50, interactive=True), gr.update(value=4.0, interactive=True)
    return gr.update(value=want, interactive=True), gr.update(value=1.0, interactive=False)


def on_aspect_change(aspect):
    size = ASPECTS.get(aspect)
    if size is None:
        return gr.update(interactive=True), gr.update(interactive=True)
    return gr.update(value=size[0], interactive=False), gr.update(value=size[1], interactive=False)


def build_ui():
    with gr.Blocks(title="Qwen-Image Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Qwen-Image Generator\n"
            "Text to image with **Qwen-Image** (20B, Apache-2.0) on a Colab GPU, "
            "powered by Hugging Face diffusers."
        )
        gr.Markdown(hardware_banner())

        with gr.Tabs():
            # ----------------------------------------------------- text2img
            with gr.Tab("Text to Image"):
                with gr.Row():
                    with gr.Column(scale=5):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder='a bookshop window at dusk, a chalkboard sign reading "OPEN LATE", warm lamplight, 35mm photograph',
                            lines=4,
                        )
                        negative = gr.Textbox(
                            label="Negative prompt (Qwen-Image really does use this)",
                            placeholder="blurry, low quality, watermark, extra fingers",
                            lines=1,
                        )
                        with gr.Row():
                            model_choice = gr.Dropdown(
                                choices=list(MODELS), value=DEFAULT_MODEL, label="Model", scale=3
                            )
                            speed_choice = gr.Dropdown(
                                choices=list(SPEED), value=DEFAULT_SPEED, label="Speed", scale=2
                            )
                        custom_repo = gr.Textbox(
                            label="Custom Hugging Face repo id",
                            placeholder="Qwen/Qwen-Image",
                            visible=False,
                        )
                        precision_choice = gr.Dropdown(
                            choices=list(PRECISION), value=DEFAULT_PRECISION, label="Precision"
                        )
                        aspect = gr.Dropdown(choices=list(ASPECTS), value=DEFAULT_ASPECT, label="Size")
                        with gr.Row():
                            width = gr.Slider(256, 2048, value=1328, step=16, label="Width", interactive=False)
                            height = gr.Slider(256, 2048, value=1328, step=16, label="Height", interactive=False)
                        with gr.Row():
                            steps = gr.Slider(1, 60, value=8, step=1, label="Steps")
                            cfg = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="True CFG scale", interactive=False)
                        with gr.Row():
                            seed = gr.Number(value=0, precision=0, label="Seed", scale=2)
                            randomize = gr.Checkbox(value=True, label="Random seed", scale=1)
                            count = gr.Slider(1, 8, value=1, step=1, label="Images", scale=2)
                        with gr.Accordion("Advanced", open=False):
                            magic = gr.Checkbox(
                                value=True,
                                label='Append Qwen\'s quality suffix ("Ultra HD, 4K, cinematic composition")',
                            )
                            max_seq = gr.Slider(
                                64, 1024, value=512, step=64,
                                label="Max prompt tokens - raise it for very long prompts",
                            )
                            gr.Markdown(
                                "**Speed** swaps in a Lightning LoRA together with the scheduler it was "
                                "distilled with. Lightning has no classifier-free guidance, so True CFG "
                                "is pinned to 1.0 and the negative prompt stops having an effect - "
                                "switch to *Quality* if you need it.\n\n"
                                "The Chinese quality suffix is used automatically when the prompt "
                                "contains Chinese or Japanese characters."
                            )
                            free_btn = gr.Button("Unload model / free VRAM", variant="secondary")
                        run = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=6):
                        gallery = gr.Gallery(
                            label="Result", columns=2, height=620, object_fit="contain",
                            preview=True, show_download_button=True,
                        )
                        status = gr.Markdown(
                            "Ready. The first run downloads the weights (~57 GB, or ~28 GB for the "
                            "NF4 model) - give it a few minutes."
                        )
                        download = gr.File(label="Download (PNG / ZIP)")

                gr.Examples(examples=[[p] for p in EXAMPLE_PROMPTS], inputs=[prompt], label="Example prompts")

            # ------------------------------------------------------ img2img
            with gr.Tab("Image to Image"):
                gr.Markdown(
                    "Redraw an existing image with the model already loaded on the first tab. "
                    "**Strength** is how far it may drift: 0.3 is a light retouch, 0.85 is almost a new image."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        i2i_image = gr.Image(label="Starting image", type="pil", height=320)
                        i2i_prompt = gr.Textbox(label="Prompt", lines=3)
                        i2i_negative = gr.Textbox(label="Negative prompt", lines=1)
                        i2i_strength = gr.Slider(0.05, 1.0, value=0.65, step=0.05, label="Strength")
                        with gr.Row():
                            i2i_steps = gr.Slider(1, 60, value=8, step=1, label="Steps")
                            i2i_cfg = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="True CFG scale")
                        with gr.Row():
                            i2i_seed = gr.Number(value=0, precision=0, label="Seed", scale=2)
                            i2i_random = gr.Checkbox(value=True, label="Random seed", scale=1)
                        i2i_magic = gr.Checkbox(value=True, label="Append the quality suffix")
                        i2i_maxseq = gr.Slider(64, 1024, value=512, step=64, label="Max prompt tokens")
                        i2i_run = gr.Button("Transform", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        i2i_gallery = gr.Gallery(label="Result", columns=1, height=620, object_fit="contain")
                        i2i_status = gr.Markdown("Generate once on the first tab so a model is in memory.")
                        i2i_download = gr.File(label="Download PNG")

            # --------------------------------------------------------- lora
            with gr.Tab("LoRA"):
                gr.Markdown(
                    "Load a Qwen-Image LoRA from the Hub (or a local `.safetensors` path) on top of "
                    "the loaded model. It stacks with the Lightning LoRA, so *Fast* and *Turbo* keep "
                    "working.\n\n"
                    "Generate one image on the first tab first, so a base model is in memory."
                )
                lora_source = gr.Textbox(
                    label="LoRA repo id or path", placeholder="flymy-ai/qwen-image-realism-lora"
                )
                lora_weight = gr.Textbox(label="Weight file (optional)", placeholder="lora.safetensors")
                lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA scale")
                with gr.Row():
                    lora_apply = gr.Button("Apply LoRA", variant="primary")
                    lora_clear = gr.Button("Remove LoRA", variant="secondary")
                lora_status = gr.Markdown("")

            # --------------------------------------------------------- help
            with gr.Tab("Help"):
                gr.Markdown(
                    f"""
### What Qwen-Image is good at

* **Text inside images** - this is its headline trick, in English *and* Chinese. Put the
  words in quotes: *a sign reading "OPEN LATE"*. It handles paragraphs, posters and slide
  layouts far better than most models.
* Long, descriptive prompts in **plain sentences**, in either language.
* It is a **true CFG** model, so unlike FLUX the negative prompt actually does something -
  but only in *Quality* mode.

### Speed and Precision

`Speed` picks how many denoising steps to run. *Quality* is the plain 50-step model.
*Fast* and *Turbo* load a Lightning LoRA distilled to 8 or 4 steps - roughly 6-12x quicker,
and close to the base model except on dense small text and fine hair detail.

`Precision` decides how 57 GB of bf16 weights fit on your card. Rough numbers at
1328x1328:

| GPU | Auto picks | 50-step Quality | 8-step Fast |
|---|---|---|---|
| A100 80GB / H100 | bf16 full GPU | ~40-60 s | ~8-12 s |
| **A100 40GB** | 4-bit NF4 | ~1.5-2 min | ~15-25 s |
| L4 24GB | 4-bit NF4 | ~5-7 min | ~50-70 s |
| T4 16GB | 4-bit NF4 | ~15-20 min | ~2-3 min |

A 40 GB A100 cannot hold the bf16 transformer (38 GiB) plus activations, so Auto drops it
to NF4 rather than letting it fail. **bf16 + group offload** is the way to get full bf16
quality on that card - it streams the transformer's layers on and off the GPU - but it is
slow, needs a high-RAM runtime, and is experimental: if it errors, fall back to NF4.

The **first** run also downloads the weights and warms up CUDA kernels. That is a one-off
per session, not per image. The *Qwen-Image NF4* model is the same weights pre-quantised,
so it downloads 28 GB instead of 57 GB.

### Files

Every image is written to `{OUTPUT_DIR}` with its prompt, seed and settings in the PNG
metadata, so the settings can be recovered later. In Colab open the folder icon in the left
sidebar to browse or download them - they are lost when the runtime disconnects.

### Licence

Qwen-Image is **Apache-2.0** - commercial use is allowed, no licence click and no token
needed. The Lightning LoRAs are Apache-2.0 too. You are still responsible for what you
generate.
                    """
                )

        # -- wiring ---------------------------------------------------------
        t2i_inputs = [prompt, negative, model_choice, custom_repo, precision_choice, speed_choice,
                      aspect, width, height, steps, cfg, seed, randomize, count, magic, max_seq]
        t2i_outputs = [gallery, status, download, seed]

        model_choice.change(on_model_change, inputs=[model_choice], outputs=[custom_repo])
        speed_choice.change(on_speed_change, inputs=[speed_choice], outputs=[steps, cfg])
        aspect.change(on_aspect_change, inputs=[aspect], outputs=[width, height])

        run.click(generate, inputs=t2i_inputs, outputs=t2i_outputs)
        prompt.submit(generate, inputs=t2i_inputs, outputs=t2i_outputs)
        free_btn.click(free_memory, outputs=[status])

        i2i_run.click(
            generate_img2img,
            inputs=[i2i_prompt, i2i_negative, i2i_image, i2i_strength, i2i_steps, i2i_cfg,
                    i2i_seed, i2i_random, i2i_magic, i2i_maxseq],
            outputs=[i2i_gallery, i2i_status, i2i_download, i2i_seed],
        )

        lora_apply.click(
            RUNNER.apply_lora, inputs=[lora_source, lora_weight, lora_scale], outputs=[lora_status]
        )
        lora_clear.click(RUNNER.remove_lora, outputs=[lora_status])

    return demo


if __name__ == "__main__":
    gpu, vram = gpu_name_and_vram()
    print(f"GPU: {gpu} ({vram:.0f} GB)")
    print(f"HF token: {'found' if hf_token() else 'none - fine, Qwen-Image is ungated'}")
    print(f"Output folder: {OUTPUT_DIR}")
    # allowed_paths: OUTPUT_DIR sits outside the working directory, and Gradio refuses to
    # serve files from anywhere it was not told about, so the download buttons need it.
    build_ui().queue(max_size=12).launch(
        share=True,
        show_error=True,
        inline=False,
        allowed_paths=[str(OUTPUT_DIR)],
    )
