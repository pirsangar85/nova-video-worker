FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    runpod \
    diffusers>=0.32.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    safetensors \
    sentencepiece \
    imageio[ffmpeg] \
    imageio-ffmpeg

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
