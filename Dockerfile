FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=2000

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN pip3 config set global.timeout 2000 && \
    pip3 config set global.retries 10 && \
    pip3 install --no-cache-dir \
        torch==2.8.0+cu128 \
        torchaudio==2.8.0+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --no-cache-dir \
        omnivoice \
        soundfile

RUN mkdir -p /output /ref_audio /root/.cache/huggingface

ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_CACHE=/root/.cache/huggingface/hub
ENV HF_ENDPOINT=https://huggingface.co
ENV HF_HUB_DOWNLOAD_TIMEOUT=3600
ENV OUTPUT_DIR=/output
ENV REF_AUDIO_DIR=/ref_audio

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8001

CMD ["python3", "/app/app.py"]
