"""
NOVA Video Generator — Dual Model Pod Worker
Fast mode: LTX-Video (5-10 sec)
Quality mode: Wan2.1-T2V (2-3 min)
Image-to-Video: Wan2.1-I2V
"""

import torch
import base64
import tempfile
import os
import traceback
import sys
import gc
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

print("=== NOVA Worker Starting ===", flush=True)
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

    # torch.compile for 30-50% speedup
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        target = getattr(pipe, 'unet', None) or getattr(pipe, 'transformer', None)
        if target is not None:
            compiled = torch.compile(target, mode="reduce-overhead")
            if hasattr(pipe, 'unet'):
                pipe.unet = compiled
            else:
                pipe.transformer = compiled
            print(f"torch.compile applied!", flush=True)
    except Exception as e:
        print(f"torch.compile skipped: {e}", flush=True)

    models[model_type] = pipe
    current_model = model_type
    print(f"{model_type} model loaded!", flush=True)
    return pipe


def generate_video(inp):
    try:
        from diffusers.utils import export_to_video
        from PIL import Image
        import io

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
            num_frames = min(inp.get("length", 121), 161)
            steps = inp.get("steps", 25)
            cfg = inp.get("cfg", 6.0)

            # Decode image
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
            num_frames = min(inp.get("length", 121), 161)
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
            num_frames = min(inp.get("length", 121), 257)
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


# Job tracking
jobs = {}
job_counter = 0
job_lock = threading.Lock()


class NovaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready", "mode": current_model or "none"}).encode())
            return

        if self.path.startswith("/status/"):
            job_id = self.path.split("/status/")[1]
            with job_lock:
                job = jobs.get(job_id)
            if job is None:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Job not found"}).encode())
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(job).encode())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/run":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            inp = data.get("input", {})

            global job_counter
            with job_lock:
                job_counter += 1
                job_id = f"nova-{job_counter}"
                jobs[job_id] = {"id": job_id, "status": "IN_QUEUE"}

            def run_job():
                with job_lock:
                    jobs[job_id]["status"] = "IN_PROGRESS"
                result = generate_video(inp)
                with job_lock:
                    if "error" in result:
                        jobs[job_id]["status"] = "FAILED"
                        jobs[job_id]["error"] = result["error"]
                    else:
                        jobs[job_id]["status"] = "COMPLETED"
                        jobs[job_id]["output"] = result

            threading.Thread(target=run_job, daemon=True).start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"id": job_id, "status": "IN_QUEUE"}).encode())
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


# Pre-load fast model
print("Pre-loading fast model...", flush=True)
load_model("fast")

print("Starting HTTP server on port 8000...", flush=True)
server = HTTPServer(("0.0.0.0", 8000), NovaHandler)
print("NOVA Worker ready! Fast + Quality + I2V modes on port 8000", flush=True)
server.serve_forever()
