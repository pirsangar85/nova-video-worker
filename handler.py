import runpod
import base64
import tempfile
import os
import traceback
import sys

print("=== NOVA Worker Starting ===", flush=True)
print(f"Python: {sys.version}", flush=True)

# Test imports early so we see errors in logs
try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"PyTorch import failed: {e}", flush=True)

try:
    import diffusers
    print(f"Diffusers: {diffusers.__version__}", flush=True)
except Exception as e:
    print(f"Diffusers import failed: {e}", flush=True)

pipe = None
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CACHE_DIR = "/runpod-volume/models" if os.path.exists("/runpod-volume") else "/tmp/models"


def load_model():
    global pipe
    if pipe is not None:
        return pipe

    import torch
    from diffusers import DiffusionPipeline

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Loading {MODEL_ID} to {CACHE_DIR}...", flush=True)

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    pipe.to("cuda")

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    print("Model loaded!", flush=True)
    return pipe


def handler(job):
    global pipe
    try:
        import torch
        from diffusers.utils import export_to_video

        inp = job["input"]
        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}

        negative_prompt = inp.get("negative_prompt", "blurry, low quality, distorted, watermark")
        width = inp.get("width", 480)
        height = inp.get("height", 832)
        num_frames = inp.get("length", 81)
        steps = inp.get("steps", 30)
        cfg = inp.get("cfg", 6.0)
        seed = inp.get("seed", -1)

        print(f"Job: '{prompt[:50]}' {width}x{height} f={num_frames} s={steps} cfg={cfg}", flush=True)

        model = load_model()
        generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

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
        print(f"Done! {len(video_b64)} chars", flush=True)

        return {"video": video_b64}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


print("Starting RunPod handler...", flush=True)
runpod.serverless.start({"handler": handler})
