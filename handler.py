"""
NOVA Video Generator — RunPod Serverless Worker
Wan2.1 Text-to-Video (14B)
Costs ~$0.01-0.03 per video (GPU time only)
"""

import runpod
import torch
import base64
import tempfile
import os

pipe = None
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B"
CACHE_DIR = "/runpod-volume/models" if os.path.exists("/runpod-volume") else "/tmp/models"

def load_model():
    global pipe
    if pipe is not None:
        return pipe

    from diffusers import WanPipeline

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Loading {MODEL_ID} (cache: {CACHE_DIR})...")

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=CACHE_DIR
    )
    pipe.to("cuda")
    pipe.enable_vae_slicing()
    print("Model loaded!")
    return pipe


def handler(job):
    global pipe
    try:
        inp = job["input"]
        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}

        negative_prompt = inp.get("negative_prompt", "blurry, low quality, distorted, watermark, text, logo, ugly, deformed")
        width = inp.get("width", 480)
        height = inp.get("height", 832)
        num_frames = inp.get("length", 81)
        steps = inp.get("steps", 40)
        cfg = inp.get("cfg", 6.0)
        seed = inp.get("seed", -1)

        model = load_model()
        from diffusers.utils import export_to_video

        generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

        print(f"Generating: '{prompt[:60]}' {width}x{height} frames={num_frames} steps={steps} cfg={cfg}")

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

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        export_to_video(output.frames[0], video_path, fps=24)

        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(video_path)

        return {"video": video_b64}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
