FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
