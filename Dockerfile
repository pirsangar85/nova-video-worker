FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install all dependencies during build — never install at runtime
RUN pip install --no-cache-dir \
    diffusers==0.34.0 \
    transformers==4.48.0 \
    accelerate==1.13.0 \
    safetensors==0.7.0 \
    sentencepiece==0.2.1 \
    ftfy==6.3.1 \
    imageio==2.37.3 \
    imageio-ffmpeg==0.6.0 \
    opencv-python-headless==4.13.0.92 \
    Pillow

# Copy handler
COPY handler.py /handler.py

# Auto-start: handler runs automatically when container starts
CMD ["bash", "-c", "HF_HUB_DISABLE_XET=1 python3 -u /handler.py"]
