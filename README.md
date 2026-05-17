# Price Tag Recognition

- build and push docker image with:

```bash
docker buildx build --platform linux/amd64 -t zarus03/price-tag-recognition:latest --push .
```

- launch the demo with:

```bash
python price_tag_recognition/demo_track.py \
    --video_path data/Данные/25_12-20/25_12-20.mp4 \
    --ckpt weights/yolo/weights/best.pt \
    --out_path result.mp4 \
    --rotate \
    --use_byte
```
