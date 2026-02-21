#!/usr/bin/env python3
"""Download IndexTTS2 model files from ModelScope"""

import requests
import os
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file with progress bar"""
    print(f"Downloading {os.path.basename(output_path)}...")
    print(f"URL: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

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
    print(f"  Size: {total_size / (1024**3):.2f} GB\n")

# ModelScope direct download URLs
# Format: https://www.modelscope.cn/models/IndexTeam/IndexTTS-2/resolve/master/{filename}
base_url = "https://www.modelscope.cn/models/IndexTeam/IndexTTS-2/resolve/master"

files_to_download = [
    ("gpt.pth", "temp_model/gpt.pth"),
    ("s2mel.pth", "temp_model/s2mel.pth"),
]

os.makedirs("temp_model", exist_ok=True)

for filename, output_path in files_to_download:
    url = f"{base_url}/{filename}"
    try:
        download_file(url, output_path)
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        # Try alternative URL format
        alt_url = f"https://www.modelscope.cn/api/v1/models/IndexTeam/IndexTTS-2/repo?Revision=master&FilePath={filename}"
        print(f"  Trying alternative URL...")
        try:
            download_file(alt_url, output_path)
        except Exception as e2:
            print(f"  ✗ Alternative URL also failed: {e2}")

print("\nDownload complete!")
