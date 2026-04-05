"""
NOVA Video Generator — RunPod Serverless Worker
Fast mode: LTX-Video (~4s clips)
Quality mode: Wan2.1-T2V (~3.5s clips)
Image-to-Video: Wan2.1-I2V
"""

import torch
import base64
import tempfile
import os
import traceback
import sys
import gc
import runpod

print("=== NOVA Serverless Worker Starting ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
try:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', 0)
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram / 1e9:.1f}GB", flush=True)
except Exception:
    pass

# Model registry
models = {}
FAST_MODEL = "Lightricks/LTX-Video-0.9.1"
QUALITY_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
I2V_MODEL = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
CACHE_DIR = "/workspace/models"
current_model = None


def unload_all():
    global models, current_model
    for name in list(models.keys()):
        del models[name]
    models = {}
    current_model = None
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_type):
    global models, current_model

    if model_type in models and models[model_type] is not None:
        return models[model_type]

    # Unload other models to free VRAM
    unload_all()
    os.makedirs(CACHE_DIR, exist_ok=True)

    if model_type == "fast":
        from diffusers import LTXPipeline
        print(f"Loading FAST: {FAST_MODEL}...", flush=True)
        pipe = LTXPipeline.from_pretrained(FAST_MODEL, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    elif model_type == "quality":
        from diffusers import DiffusionPipeline
        print(f"Loading QUALITY: {QUALITY_MODEL}...", flush=True)
        pipe = DiffusionPipeline.from_pretrained(QUALITY_MODEL, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    elif model_type == "i2v":
        from diffusers import DiffusionPipeline
        print(f"Loading I2V: {I2V_MODEL}...", flush=True)
        pipe = DiffusionPipeline.from_pretrained(I2V_MODEL, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipe.to("cuda")
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    models[model_type] = pipe
    current_model = model_type
    print(f"{model_type} model loaded!", flush=True)
    return pipe


def handler(job):
    """RunPod serverless handler"""
    try:
        from diffusers.utils import export_to_video
        from PIL import Image
        import io

        inp = job["input"]
        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}

        mode = inp.get("mode", "fast")
        negative_prompt = inp.get("negative_prompt", "blurry, low quality, distorted, watermark, text, logo, ugly, deformed")
        width = inp.get("width", 480)
        height = inp.get("height", 832)
        seed = inp.get("seed", -1)
        image_base64 = inp.get("image_base64", None)

        generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

        # If image provided, use I2V model
        if image_base64:
            mode = "i2v"
            num_frames = min(inp.get("length", 81), 81)
            steps = inp.get("steps", 25)
            cfg = inp.get("cfg", 6.0)

            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            img_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image = image.resize((width, height))

            print(f"[I2V] '{prompt[:50]}' {width}x{height} f={num_frames} s={steps}", flush=True)
            model = load_model("i2v")

            output = model(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )

        elif mode == "quality":
            num_frames = min(inp.get("length", 81), 81)
            steps = inp.get("steps", 25)
            cfg = inp.get("cfg", 6.0)

            print(f"[QUALITY] '{prompt[:50]}' {width}x{height} f={num_frames} s={steps}", flush=True)
            model = load_model("quality")

            output = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )

        else:
            num_frames = min(inp.get("length", 97), 97)
            steps = inp.get("steps", 8)
            cfg = inp.get("cfg", 3.0)

            print(f"[FAST] '{prompt[:50]}' {width}x{height} f={num_frames} s={steps}", flush=True)
            model = load_model("fast")

            output = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )

        video_path = os.path.join(tempfile.gettempdir(), f"nova_{os.getpid()}.mp4")
        export_to_video(output.frames[0], video_path, fps=24)

        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(video_path)

        del output
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Done! {len(video_b64)} chars", flush=True)
        return {"video": video_b64}

    except Exception as e:
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": str(e)}


# Pre-load fast model on cold start
print("Pre-loading fast model...", flush=True)
load_model("fast")

print("NOVA Serverless Worker ready!", flush=True)
runpod.serverless.start({"handler": handler})
