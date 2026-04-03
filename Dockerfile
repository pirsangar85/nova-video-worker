FROM runpod/base:0.6.2-cuda12.2.0

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install video generation deps
RUN pip install --no-cache-dir \
    diffusers>=0.31.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    safetensors \
    sentencepiece \
    imageio[ffmpeg] \
    imageio-ffmpeg

WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
