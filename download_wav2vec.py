#!/usr/bin/env python3
"""Download Wav2Vec-BERT 2.0 weights from Hugging Face"""

import requests
import os
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file with progress bar"""
    print(f"Downloading {os.path.basename(output_path)}...")
    print(f"URL: {url}")

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"  ✗ Failed: HTTP {response.status_code}")
        return False

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"✓ Downloaded to {output_path}")
    if total_size > 0:
        print(f"  Size: {total_size / (1024**3):.2f} GB\n")
    return True

# Wav2Vec-BERT 2.0 from Hugging Face
# This is the model used by IndexTTS2
model_id = "facebook/wav2vec-bert-2.0"

# Files to download
files = [
    "model.safetensors",
    "config.json",
]

base_url = f"https://huggingface.co/{model_id}/resolve/main"

print(f"Downloading Wav2Vec-BERT 2.0 from Hugging Face...")
print(f"Model: {model_id}\n")

for filename in files:
    url = f"{base_url}/{filename}"
    output_path = f"checkpoints/wav2vec_bert/{filename}"
    download_file(url, output_path)

print("\nDownload complete!")
print("\nNote: The model weights need to be converted to the format expected by IndexTTS2.")
print("Please check the documentation for conversion instructions.")
