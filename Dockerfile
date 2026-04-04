FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    diffusers>=0.33.0 \
    transformers>=4.47.0 \
    accelerate>=0.34.0 \
    safetensors \
    sentencepiece \
    imageio[ffmpeg] \
    imageio-ffmpeg \
    Pillow

EXPOSE 8000

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
