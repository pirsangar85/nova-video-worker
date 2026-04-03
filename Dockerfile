FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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
