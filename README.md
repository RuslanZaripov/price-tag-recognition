# Price Tag Recognition

build and push docker image with:

```bash
docker buildx build --platform linux/amd64 -t zarus03/price-tag-recognition:latest --push .
```

launch the demo with:

```bash
python demo_track.py
```
