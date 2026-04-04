FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    diffusers==0.32.2 \
    transformers==4.47.0 \
    accelerate==0.34.2 \
    safetensors==0.4.5 \
    sentencepiece==0.2.0 \
    imageio[ffmpeg]==2.36.1 \
    imageio-ffmpeg==0.5.1 \
    Pillow==11.0.0

# Force cache bust v2
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
