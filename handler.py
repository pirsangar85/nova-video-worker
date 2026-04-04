"""
NOVA Video Generator — RunPod Pod Worker
Wan2.1 Text-to-Video (1.3B)
Runs as HTTP server on port 8000
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
        vram = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram / 1e9:.1f}GB", flush=True)
except Exception as e:
    print(f"GPU info error (non-fatal): {e}", flush=True)

pipe = None
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CACHE_DIR = "/workspace/models"


def load_model():
    global pipe
    if pipe is not None:
        return pipe

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


def generate_video(inp):
    try:
        from diffusers.utils import export_to_video

        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}

        negative_prompt = inp.get("negative_prompt", "blurry, low quality, distorted, watermark, text, logo, ugly, deformed")
        width = inp.get("width", 480)
        height = inp.get("height", 832)
        num_frames = min(inp.get("length", 81), 81)
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


# Job tracking for async mode
jobs = {}
job_counter = 0
job_lock = threading.Lock()


class NovaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready"}).encode())
            return

        # GET /status/{job_id}
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

            # Run in background thread
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
        pass  # Suppress default logging


# Pre-load model on startup
print("Pre-loading model...", flush=True)
load_model()

print("Starting HTTP server on port 8000...", flush=True)
server = HTTPServer(("0.0.0.0", 8000), NovaHandler)
print("NOVA Worker ready! Listening on port 8000", flush=True)
server.serve_forever()
