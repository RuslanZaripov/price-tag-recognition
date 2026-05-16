import json
import torch
import requests
from peft import PeftModel
from price_tag_recognition.parse_json import extract_json
from transformers import AutoProcessor, AutoModelForImageTextToText


def initialize_vlm(device="cpu"):
    base_id = "Qwen/Qwen3-VL-8B-Instruct"
    lora_id = "openfoodfacts/price-tag-extractor"

    processor = AutoProcessor.from_pretrained(base_id)

    max_memory = {0: "14GiB", 1: "14GiB", "cpu": "30GiB"}

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_id, torch_dtype=torch.float16, device_map="auto", max_memory=max_memory
    )

    model = PeftModel.from_pretrained(base_model, lora_id, autocast_adapter_dtype=False)

    model.eval()

    return model, processor


def get_prompt():
    config = requests.get(
        "https://huggingface.co/datasets/openfoodfacts/price-tag-extraction/resolve/v1.1/config.json",
    ).json()
    json_schema = config["json_schema"]
    instructions = config["instructions"]
    json_schema_str = json.dumps(json_schema)
    full_instructions = f"{instructions}\n\nResponse must be formatted as JSON, and follow this JSON schema:\n{json_schema_str}"
    return full_instructions


def run_vlm_batch(images, model, processor, batch_size=8):
    results = []

    full_instructions = get_prompt()

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]

        # Build messages per image
        messages_batch = []
        for img in batch_imgs:
            messages_batch.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": full_instructions},
                        ],
                    }
                ]
            )

        # Build prompts
        prompts = [
            processor.apply_chat_template(m, add_generation_prompt=True)
            for m in messages_batch
        ]

        # Tokenize (BATCH!)
        inputs = processor(
            text=prompts, images=batch_imgs, return_tensors="pt", padding=True
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)

        parsed_batch = []

        for text in decoded:
            parsed = list(extract_json(text))
            parsed_batch.append(parsed if parsed else text)

        results.extend(parsed_batch)

    return results
