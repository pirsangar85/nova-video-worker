FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    diffusers>=0.31.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    safetensors \
    sentencepiece \
    runpod \
    imageio[ffmpeg] \
    imageio-ffmpeg

WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
