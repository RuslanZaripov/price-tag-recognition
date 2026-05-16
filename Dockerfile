FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

COPY . .

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1

RUN pip install --no-cache-dir -r /workspace/requirements.txt

RUN chmod +x /workspace/entrypoint.sh

CMD [ "/bin/bash" ]
