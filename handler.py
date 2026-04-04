"""
NOVA Video Generator — Dual Model Pod Worker
Fast mode: LTX-Video (5-10 sec)
Quality mode: Wan2.1-T2V (2-3 min)
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

fast_pipe = None
quality_pipe = None
FAST_MODEL = "Lightricks/LTX-Video-0.9.1"
QUALITY_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CACHE_DIR = "/workspace/models"
current_mode = None  # Track which model is loaded to manage VRAM


def load_fast_model():
    global fast_pipe, quality_pipe, current_mode
    if fast_pipe is not None:
        return fast_pipe

    # Unload quality model to free VRAM
    if quality_pipe is not None:
        print("Unloading quality model...", flush=True)
        quality_pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    from diffusers import LTXPipeline

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Loading FAST model: {FAST_MODEL}...", flush=True)

    fast_pipe = LTXPipeline.from_pretrained(
        FAST_MODEL,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    fast_pipe.to("cuda")
    try:
        fast_pipe.enable_vae_slicing()
    except Exception:
        pass

    current_mode = "fast"
    print("Fast model loaded!", flush=True)
    return fast_pipe


def load_quality_model():
    global fast_pipe, quality_pipe, current_mode
    if quality_pipe is not None:
        return quality_pipe

    # Unload fast model to free VRAM
    if fast_pipe is not None:
        print("Unloading fast model...", flush=True)
        fast_pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    from diffusers import DiffusionPipeline

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Loading QUALITY model: {QUALITY_MODEL}...", flush=True)

    quality_pipe = DiffusionPipeline.from_pretrained(
        QUALITY_MODEL,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    quality_pipe.to("cuda")
    try:
        quality_pipe.enable_vae_slicing()
    except Exception:
        pass

    current_mode = "quality"
    print("Quality model loaded!", flush=True)
    return quality_pipe


def generate_video(inp):
    try:
        from diffusers.utils import export_to_video

        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}

        mode = inp.get("mode", "fast")  # "fast" or "quality"
        negative_prompt = inp.get("negative_prompt", "blurry, low quality, distorted, watermark, text, logo, ugly, deformed")
        width = inp.get("width", 480)
        height = inp.get("height", 832)
        seed = inp.get("seed", -1)

        generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

        if mode == "quality":
            # Wan2.1 — high quality, slower
            num_frames = min(inp.get("length", 81), 81)
            steps = inp.get("steps", 25)
            cfg = inp.get("cfg", 6.0)

            print(f"[QUALITY] '{prompt[:50]}' {width}x{height} f={num_frames} s={steps}", flush=True)
            model = load_quality_model()

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
            # LTX-Video — fast mode
            num_frames = min(inp.get("length", 49), 97)
            steps = inp.get("steps", 8)
            cfg = inp.get("cfg", 3.0)

            print(f"[FAST] '{prompt[:50]}' {width}x{height} f={num_frames} s={steps}", flush=True)
            model = load_fast_model()

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
            self.wfile.write(json.dumps({"status": "ready", "mode": current_mode}).encode())
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


# Pre-load fast model (most common)
print("Pre-loading fast model...", flush=True)
load_fast_model()

print("Starting HTTP server on port 8000...", flush=True)
server = HTTPServer(("0.0.0.0", 8000), NovaHandler)
print("NOVA Worker ready! Fast + Quality modes available on port 8000", flush=True)
server.serve_forever()
