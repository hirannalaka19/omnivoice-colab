"""FLUX.1 image generator - Gradio app, tuned for Google Colab (A100 friendly).

Wraps black-forest-labs/FLUX.1-dev (and its siblings) in a small Gradio UI so a
Colab notebook can become a text-to-image studio with a public link.

FLUX.1-dev is a gated Hugging Face repo: accept the licence once at
https://huggingface.co/black-forest-labs/FLUX.1-dev and put a read token in the
HF_TOKEN environment variable (the notebook reads it from Colab Secrets).
"""

import gc
import os
import random
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
OUTPUT_DIR = Path("/content/Flux_Output") if Path("/content").is_dir() else Path("Flux_Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEED = 2**31 - 1

MODELS = {
    "FLUX.1-dev  -  best quality (gated)": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "steps": 28,
        "guidance": 3.5,
        "max_seq": 512,
    },
    "FLUX.1-Krea-dev  -  photographic look (gated)": {
        "repo": "black-forest-labs/FLUX.1-Krea-dev",
        "steps": 28,
        "guidance": 4.5,
        "max_seq": 512,
    },
    "FLUX.1-schnell  -  4 steps, Apache-2.0 (open)": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "steps": 4,
        "guidance": 0.0,
        "max_seq": 256,
    },
    "Custom repo id...": {
        "repo": "",
        "steps": 28,
        "guidance": 3.5,
        "max_seq": 512,
    },
}
DEFAULT_MODEL = list(MODELS)[0]
CUSTOM_MODEL = "Custom repo id..."

# FLUX needs both sides divisible by 16 (VAE downscales 8x, then 2x2 patches).
ASPECTS = {
    "1:1  square         1024 x 1024": (1024, 1024),
    "3:2  landscape      1216 x 832": (1216, 832),
    "2:3  portrait        832 x 1216": (832, 1216),
    "4:3  landscape      1152 x 896": (1152, 896),
    "3:4  portrait        896 x 1152": (896, 1152),
    "16:9 wide           1344 x 768": (1344, 768),
    "9:16 phone           768 x 1344": (768, 1344),
    "21:9 cinematic      1536 x 640": (1536, 640),
    "Custom (use the sliders below)": None,
}
DEFAULT_ASPECT = list(ASPECTS)[0]

PRECISION = {
    "Auto  -  pick from the detected GPU": "auto",
    "bf16 full GPU  -  fastest, needs ~34 GB (A100 40GB, H100)": "bf16",
    "bf16 + CPU offload  -  ~20 GB VRAM (L4, A10, V100 32GB)": "offload",
    "4-bit NF4  -  ~11 GB VRAM, slower (T4, free tier)": "nf4",
}
DEFAULT_PRECISION = list(PRECISION)[0]

EXAMPLE_PROMPTS = [
    "a golden retriever puppy asleep on a stack of old books, warm window light, 85mm photograph, shallow depth of field",
    "cinematic still of a lone lighthouse in a storm, huge waves, moody blue hour, anamorphic lens flare, 35mm film grain",
    "an isometric miniature diorama of a cozy ramen shop at night, tilt-shift, soft neon signage, highly detailed",
    "studio product shot of a matte black ceramic coffee mug on polished concrete, softbox lighting, minimal, editorial",
    "a watercolour illustration of Kandy lake at sunrise, misty hills, loose brush strokes, pastel palette",
    "portrait of an elderly fisherman, weathered face, rim light, shot on Kodak Portra 400, natural skin texture",
    'a hand-painted wooden shop sign that reads "OPEN EARLY", morning sun, peeling paint, macro detail',
]


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
    """Pick a loading strategy that fits the GPU we were actually handed."""
    _, total = gpu_name_and_vram()
    if total >= 34:      # A100 40/80GB, H100 - the whole pipeline stays resident
        return "bf16"
    if total >= 20:      # L4 24GB, A10, V100 32GB - swap modules in and out
        return "offload"
    return "nf4"         # T4 16GB and friends - quantise the two big blocks


def hardware_banner():
    name, total = gpu_name_and_vram()
    if total == 0:
        return (
            "### No GPU detected\n"
            "In Colab: **Runtime -> Change runtime type -> GPU -> A100** "
            "(A100 needs Colab Pro; L4 and T4 also work, just slower)."
        )
    picked = [k for k, v in PRECISION.items() if v == auto_precision()][0]
    tier = "plenty of headroom" if total >= 34 else ("workable" if total >= 20 else "tight, expect slow runs")
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


def gated_repo_error(repo, err):
    """Turn a 401/403 from the Hub into instructions the user can act on."""
    return gr.Error(
        f"Cannot download '{repo}' - it is a gated repo.\n\n"
        f"1. Open https://huggingface.co/{repo} while signed in and accept the licence.\n"
        "2. Create a READ token at https://huggingface.co/settings/tokens\n"
        "3. In Colab click the key icon in the left sidebar -> Add new secret -> "
        "name HF_TOKEN, paste the token, enable Notebook access.\n"
        "4. Re-run the notebook cell so the app picks the token up.\n\n"
        f"(underlying error: {type(err).__name__}: {str(err)[:300]})"
    )


# --------------------------------------------------------------------------
# Pipeline manager
# --------------------------------------------------------------------------

class FluxRunner:
    """Holds one loaded FLUX pipeline and reuses it across generations."""

    def __init__(self):
        self.pipe = None
        self.i2i = None
        self.repo = None
        self.mode = None
        self.lora = None

    # -- memory ------------------------------------------------------------

    def unload(self):
        self.pipe = None
        self.i2i = None
        self.repo = None
        self.mode = None
        self.lora = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # -- loading -----------------------------------------------------------

    def _load_nf4(self, repo, token):
        """4-bit the two heavyweights (12B transformer, 4.7B T5) so a 16 GB card copes."""
        from diffusers import BitsAndBytesConfig as DiffusersBnb
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig as TransformersBnb
        from transformers import T5EncoderModel

        transformer = FluxTransformer2DModel.from_pretrained(
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
        text_encoder_2 = T5EncoderModel.from_pretrained(
            repo,
            subfolder="text_encoder_2",
            quantization_config=TransformersBnb(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
            token=token,
        )
        pipe = FluxPipeline.from_pretrained(
            repo,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        # bitsandbytes modules cannot be moved with .to(), so offload hooks are the way.
        pipe.enable_model_cpu_offload()
        return pipe

    def load(self, repo, mode, progress=None):
        if not repo:
            raise gr.Error("Pick a model, or type a Hugging Face repo id in the custom field.")
        if not torch.cuda.is_available():
            raise gr.Error("No GPU. In Colab: Runtime -> Change runtime type -> GPU (A100 recommended).")

        if mode == "auto":
            mode = auto_precision()
        if self.pipe is not None and self.repo == repo and self.mode == mode:
            return mode

        self.unload()
        token = hf_token()
        if progress:
            progress(0.05, desc=f"Downloading {repo} - first run pulls ~34 GB, a few minutes...")

        try:
            if mode == "nf4":
                pipe = self._load_nf4(repo, token)
            else:
                from diffusers import FluxPipeline

                pipe = FluxPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, token=token)
                if mode == "offload":
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to("cuda")
        except Exception as err:  # noqa: BLE001 - re-raised as a readable gr.Error
            text = f"{type(err).__name__} {err}".lower()
            if any(s in text for s in ("gated", "401", "403", "awaiting a review", "restricted", "unauthorized")):
                raise gated_repo_error(repo, err)
            if "out of memory" in text:
                raise gr.Error(
                    "Ran out of VRAM while loading. Set Precision to "
                    "'bf16 + CPU offload' or '4-bit NF4' and try again."
                )
            raise gr.Error(f"Could not load '{repo}': {type(err).__name__}: {str(err)[:400]}")

        pipe.set_progress_bar_config(disable=True)
        # Slicing is free. Tiling is only worth it on a small card: it kicks in above
        # 1 megapixel and can leave faint seams, which an A100 has no reason to risk.
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        if mode == "nf4":
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass

        self.pipe = pipe
        self.repo = repo
        self.mode = mode
        self.i2i = None
        self.lora = None
        return mode

    def img2img(self):
        """Image-to-image pipeline sharing the loaded weights (costs no extra VRAM)."""
        if self.pipe is None:
            raise gr.Error("Load a model first - generate one image on the 'Text -> Image' tab.")
        if self.i2i is None:
            from diffusers import FluxImg2ImgPipeline

            # from_pipe reuses the same modules and keeps any offload hooks.
            self.i2i = FluxImg2ImgPipeline.from_pipe(self.pipe)
            self.i2i.set_progress_bar_config(disable=True)
        return self.i2i

    # -- LoRA --------------------------------------------------------------

    def apply_lora(self, source, weight_name, scale):
        if self.pipe is None:
            raise gr.Error("Load a model first (generate one image), then apply a LoRA.")
        source = (source or "").strip()
        if not source:
            raise gr.Error("Enter a LoRA repo id (e.g. XLabs-AI/flux-RealismLora) or a .safetensors path.")
        self.remove_lora()
        kwargs = {"adapter_name": "user_lora"}
        if (weight_name or "").strip():
            kwargs["weight_name"] = weight_name.strip()
        token = hf_token()
        if token:
            kwargs["token"] = token
        try:
            self.pipe.load_lora_weights(source, **kwargs)
            self.pipe.set_adapters(["user_lora"], adapter_weights=[float(scale)])
        except Exception as err:  # noqa: BLE001
            raise gr.Error(f"LoRA failed to load: {type(err).__name__}: {str(err)[:300]}")
        self.lora = source
        self.i2i = None  # rebuild so the img2img pipe sees the adapter
        return f"LoRA active: `{source}` at scale {float(scale):.2f}"

    def remove_lora(self):
        if self.pipe is None:
            return "Nothing loaded yet."
        try:
            self.pipe.unload_lora_weights()
        except Exception:
            pass
        self.lora = None
        self.i2i = None
        return "LoRA removed - back to the base model."


RUNNER = FluxRunner()


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

def save_image(image, prompt, seed, steps, guidance, repo, size):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"flux_{stamp}_{seed}.png"
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "parameters",
        f"{prompt}\nModel: {repo}, Steps: {steps}, Guidance: {guidance}, "
        f"Seed: {seed}, Size: {size[0]}x{size[1]}",
    )
    image.save(path, pnginfo=info)
    return path


def zip_paths(paths):
    if not paths:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = OUTPUT_DIR / f"flux_batch_{stamp}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, Path(p).name)
    return str(archive)


def resolve_model(choice, custom_repo):
    spec = MODELS.get(choice, MODELS[DEFAULT_MODEL])
    repo = spec["repo"] or (custom_repo or "").strip()
    return repo, spec


def snap16(value):
    """FLUX wants multiples of 16 on both axes."""
    return max(256, int(round(float(value) / 16) * 16))


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate(
    prompt,
    model_choice,
    custom_repo,
    precision_choice,
    aspect,
    width,
    height,
    steps,
    guidance,
    seed,
    randomize,
    count,
    max_seq,
    progress=gr.Progress(),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write a prompt first.")

    repo, _spec = resolve_model(model_choice, custom_repo)
    mode = PRECISION.get(precision_choice, "auto")

    progress(0.02, desc="Preparing model...")
    t_load = time.time()
    used_mode = RUNNER.load(repo, mode, progress=progress)
    load_secs = time.time() - t_load

    if ASPECTS.get(aspect):
        width, height = ASPECTS[aspect]
    width, height = snap16(width), snap16(height)

    steps = int(steps)
    count = max(1, int(count))
    guidance = float(guidance)
    max_seq = int(max_seq)
    # schnell is timestep-distilled: no guidance embedding, and T5 caps at 256 tokens.
    if "schnell" in repo.lower():
        max_seq = min(max_seq, 256)

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
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=max_seq,
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
                f"CUDA out of memory at {width}x{height}. Lower the resolution, drop the "
                "image count to 1, or set Precision to '4-bit NF4'."
            )
        took = time.time() - started

        image = result.images[0]
        path = save_image(image, prompt, this_seed, steps, guidance, repo, (width, height))
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
        f"{width}x{height} - {steps} steps - guidance {guidance} - precision `{used_mode}`  \n"
        f"GPU {name} ({vram_used():.1f} / {total:.0f} GB in use)"
        + (f"  \nModel load / warm-up: {load_secs:.0f}s" if load_secs > 5 else "")
        + f"  \nSaved to `{OUTPUT_DIR}`"
    )
    yield images, status, archive, base_seed


def generate_img2img(
    prompt,
    init_image,
    strength,
    steps,
    guidance,
    seed,
    randomize,
    max_seq,
    progress=gr.Progress(),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write a prompt describing what the result should look like.")
    if init_image is None:
        raise gr.Error("Upload a starting image.")
    if RUNNER.pipe is None:
        raise gr.Error("Load a model first - generate one image on the 'Text -> Image' tab.")

    pipe = RUNNER.img2img()
    image = init_image.convert("RGB")
    width, height = snap16(image.width), snap16(image.height)
    image = image.resize((width, height), Image.LANCZOS)

    steps = int(steps)
    strength = float(strength)
    this_seed = random.randint(0, MAX_SEED) if randomize else int(seed)
    generator = torch.Generator("cpu").manual_seed(this_seed)
    # img2img only walks the tail of the schedule, so that is all the progress bar sees.
    expected = max(1, int(steps * strength))

    def step_cb(_pipe, i, _t, kwargs):
        progress(min(1.0, (i + 1) / expected), desc=f"step {i + 1}/{expected}")
        return kwargs

    call = dict(
        prompt=prompt,
        image=image,
        strength=strength,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=float(guidance),
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
    path = save_image(out, prompt, this_seed, steps, guidance, RUNNER.repo or "", (width, height))
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
    """Move steps / guidance / token limit to sane defaults for the picked model."""
    spec = MODELS.get(choice, MODELS[DEFAULT_MODEL])
    return (
        gr.update(visible=choice == CUSTOM_MODEL),
        gr.update(value=spec["steps"]),
        gr.update(value=spec["guidance"]),
        gr.update(value=spec["max_seq"], maximum=spec["max_seq"]),
    )


def on_aspect_change(aspect):
    size = ASPECTS.get(aspect)
    if size is None:
        return gr.update(interactive=True), gr.update(interactive=True)
    return gr.update(value=size[0], interactive=False), gr.update(value=size[1], interactive=False)


def build_ui():
    with gr.Blocks(title="FLUX.1 Image Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# FLUX.1 Image Generator\nText to image on a Colab GPU, powered by Hugging Face diffusers.")
        gr.Markdown(hardware_banner())
        if not hf_token():
            gr.Markdown(
                "> **No `HF_TOKEN` found.** FLUX.1-dev is gated - accept the licence at "
                "https://huggingface.co/black-forest-labs/FLUX.1-dev, then add the token as a Colab "
                "secret named `HF_TOKEN` (key icon in the left sidebar, enable *Notebook access*) and "
                "re-run the cell. FLUX.1-schnell works without a token."
            )

        with gr.Tabs():
            # ----------------------------------------------------- text2img
            with gr.Tab("Text to Image"):
                with gr.Row():
                    with gr.Column(scale=5):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="a golden retriever puppy asleep on a stack of old books, warm window light, 85mm photograph",
                            lines=4,
                        )
                        with gr.Row():
                            model_choice = gr.Dropdown(
                                choices=list(MODELS), value=DEFAULT_MODEL, label="Model", scale=3
                            )
                            precision_choice = gr.Dropdown(
                                choices=list(PRECISION), value=DEFAULT_PRECISION, label="Precision", scale=3
                            )
                        custom_repo = gr.Textbox(
                            label="Custom Hugging Face repo id",
                            placeholder="black-forest-labs/FLUX.1-dev",
                            visible=False,
                        )
                        aspect = gr.Dropdown(choices=list(ASPECTS), value=DEFAULT_ASPECT, label="Size")
                        with gr.Row():
                            width = gr.Slider(256, 2048, value=1024, step=16, label="Width", interactive=False)
                            height = gr.Slider(256, 2048, value=1024, step=16, label="Height", interactive=False)
                        with gr.Row():
                            steps = gr.Slider(1, 60, value=28, step=1, label="Steps")
                            guidance = gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Guidance")
                        with gr.Row():
                            seed = gr.Number(value=0, precision=0, label="Seed", scale=2)
                            randomize = gr.Checkbox(value=True, label="Random seed", scale=1)
                            count = gr.Slider(1, 8, value=1, step=1, label="Images", scale=2)
                        with gr.Accordion("Advanced", open=False):
                            max_seq = gr.Slider(
                                64, 512, value=512, step=64,
                                label="Max prompt tokens (T5) - lower is slightly faster",
                            )
                            gr.Markdown(
                                "FLUX.1-dev is guidance-distilled, so there is **no negative prompt**. "
                                "Describe what you *do* want, in plain sentences - it reads long prompts "
                                "well. Guidance 2.5-4 stays natural; above 6 it gets crunchy."
                            )
                            free_btn = gr.Button("Unload model / free VRAM", variant="secondary")
                        run = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=6):
                        gallery = gr.Gallery(
                            label="Result", columns=2, height=620, object_fit="contain",
                            preview=True, show_download_button=True,
                        )
                        status = gr.Markdown(
                            "Ready. The first run downloads ~34 GB of weights - give it a few minutes."
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
                        i2i_strength = gr.Slider(0.05, 1.0, value=0.65, step=0.05, label="Strength")
                        with gr.Row():
                            i2i_steps = gr.Slider(1, 60, value=28, step=1, label="Steps")
                            i2i_guidance = gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Guidance")
                        with gr.Row():
                            i2i_seed = gr.Number(value=0, precision=0, label="Seed", scale=2)
                            i2i_random = gr.Checkbox(value=True, label="Random seed", scale=1)
                        i2i_maxseq = gr.Slider(64, 512, value=512, step=64, label="Max prompt tokens")
                        i2i_run = gr.Button("Transform", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        i2i_gallery = gr.Gallery(label="Result", columns=1, height=620, object_fit="contain")
                        i2i_status = gr.Markdown("Generate once on the first tab so a model is in memory.")
                        i2i_download = gr.File(label="Download PNG")

            # --------------------------------------------------------- lora
            with gr.Tab("LoRA"):
                gr.Markdown(
                    "Load a FLUX LoRA from the Hub (or a local `.safetensors` path) on top of the "
                    "loaded model.\n\n"
                    "Examples: `XLabs-AI/flux-RealismLora`, `alvdansen/flux-koda`, "
                    "`Shakker-Labs/FLUX.1-dev-LoRA-add-details`.\n\n"
                    "Generate one image on the first tab first, so a base model is in memory."
                )
                lora_source = gr.Textbox(label="LoRA repo id or path", placeholder="XLabs-AI/flux-RealismLora")
                lora_weight = gr.Textbox(label="Weight file (optional)", placeholder="lora.safetensors")
                lora_scale = gr.Slider(0.0, 2.0, value=0.9, step=0.05, label="LoRA scale")
                with gr.Row():
                    lora_apply = gr.Button("Apply LoRA", variant="primary")
                    lora_clear = gr.Button("Remove LoRA", variant="secondary")
                lora_status = gr.Markdown("")

            # --------------------------------------------------------- help
            with gr.Tab("Help"):
                gr.Markdown(
                    f"""
### Getting a good result

* Write **descriptive sentences**, not keyword soup - FLUX reads natural language.
* Say the medium and the light: *"35mm photograph, overcast light"*, *"flat vector illustration"*.
* FLUX is unusually good at **text inside images** - put the words in quotes.
* Same seed + same settings = the same image. Change one thing at a time.

### Speed on Colab

| GPU | Precision | ~1024x1024, 28 steps |
|---|---|---|
| A100 40GB | bf16 full GPU | ~10-15 s |
| L4 24GB | bf16 + CPU offload | ~50-70 s |
| T4 16GB | 4-bit NF4 | ~2-4 min |

The **first** run also downloads ~34 GB of weights and warms up CUDA kernels. That is a
one-off per session, not per image.

### Files

Every image is written to `{OUTPUT_DIR}` with its prompt, seed and settings stored in the
PNG metadata, so the settings can be recovered later. In Colab open the folder icon in the
left sidebar to browse or download them - they are lost when the runtime disconnects.

### Licence

FLUX.1-dev and FLUX.1-Krea-dev are released under the **FLUX.1 [dev] Non-Commercial
License** - fine for personal and research work, not for commercial use. FLUX.1-schnell is
Apache-2.0 and can be used commercially. You are responsible for what you generate.
                    """
                )

        # -- wiring ---------------------------------------------------------
        t2i_inputs = [prompt, model_choice, custom_repo, precision_choice, aspect, width, height,
                      steps, guidance, seed, randomize, count, max_seq]
        t2i_outputs = [gallery, status, download, seed]

        model_choice.change(
            on_model_change, inputs=[model_choice], outputs=[custom_repo, steps, guidance, max_seq]
        )
        aspect.change(on_aspect_change, inputs=[aspect], outputs=[width, height])

        run.click(generate, inputs=t2i_inputs, outputs=t2i_outputs)
        prompt.submit(generate, inputs=t2i_inputs, outputs=t2i_outputs)
        free_btn.click(free_memory, outputs=[status])

        i2i_run.click(
            generate_img2img,
            inputs=[i2i_prompt, i2i_image, i2i_strength, i2i_steps, i2i_guidance,
                    i2i_seed, i2i_random, i2i_maxseq],
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
    print(f"HF token: {'found' if hf_token() else 'MISSING - gated models will fail'}")
    print(f"Output folder: {OUTPUT_DIR}")
    build_ui().queue(max_size=12).launch(share=True, show_error=True, inline=False)
