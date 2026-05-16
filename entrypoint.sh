#!/bin/bash

# Cause the script to exit on failure.
set -eo pipefail

cd /workspace

data_dir="/workspace/data"
weights_dir="/workspace/weights"

# clone YOLO weights
git clone https://huggingface.co/openfoodfacts/price-tag-detection "$weights_dir/yolo"

# download data
gdown --folder --fuzzy \
    https://drive.google.com/drive/folders/1_UbQ7x4MK9fZjA-9DqJY_A8nedsg9KMy?usp=sharing \
    -O "$data_dir"

