# Base image: NVIDIA Triton Inference Server
FROM nvcr.io/nvidia/tritonserver:25.01-pyt-python-py3

RUN pip install --no-cache-dir \
    torch \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    Pillow \
    hf_transfer \
    protobuf \
    bitsandbytes \
    sentencepiece \
    numpy \
    runpod

# Copy HiDream model serving files
RUN mkdir -p /models/hidream/1
COPY models/hidream/1/model.py /models/hidream/1
COPY models/hidream/config.pbtxt /models/hidream/

# Copy handler
COPY handler.py /app/handler.py
COPY requirements.txt /app/requirements.txt

# Env vars for HF downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HUGGINGFACE_HUB_CACHE=/cache/huggingface

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start both Triton & RunPod handler
CMD ["sh", "-c", "tritonserver --model-repository=/models & python /app/handler.py"]
