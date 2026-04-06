FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git cmake build-essential \
    libsentencepiece-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ Pin numpy first to avoid numpy 2.x conflict
RUN pip3 install "numpy<2"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install git+https://github.com/openvla/openvla.git

COPY openvla_server.py .
CMD ["python3", "openvla_server.py"]