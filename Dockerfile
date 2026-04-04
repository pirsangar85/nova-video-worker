FROM runpod/base:0.6.2-cuda12.2.0

# Install PyTorch matching CUDA 12.2
RUN pip install --no-cache-dir \
    torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# Install diffusers and deps
RUN pip install --no-cache-dir \
    diffusers>=0.32.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    safetensors \
    sentencepiece \
    imageio[ffmpeg] \
    imageio-ffmpeg

# Copy handler to root
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
