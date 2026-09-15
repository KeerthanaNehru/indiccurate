#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add venv_packages to Python path
sys.path.insert(0, '/Users/keerthana.n/Documents/New_Project_Folder/Data_Curator_Project/indiccurate/venv_packages')

# Set HF token from .env
env_path = Path('.env')
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.startswith('HF_TOKEN='):
                token = line.split('=')[1].strip()
                os.environ['HF_TOKEN'] = token
                print(f"✅ HF_TOKEN loaded from .env")
                break

# Import after setting token
from huggingface_hub import HfApi

# Configuration
REPO_ID = "KeerthanaNehru/indiccurate-tamil"
REPO_TYPE = "dataset"

# Ensure token is set
if "HF_TOKEN" not in os.environ:
    raise ValueError("HF_TOKEN not set in environment!")

api = HfApi()

print(f"\n📤 Uploading to: {REPO_ID}\n")

files_to_upload = [
    ("data/processed/text_clean.parquet", "text_clean.parquet"),
    ("data/processed/audio_clean.parquet", "audio_clean.parquet"),
    ("data/processed/contamination_report.json", "contamination_report.json"),
    ("data/processed/audio_qa_report.json", "audio_qa_report.json"),
]

# Upload individual files
for local_path, repo_path in files_to_upload:
    try:
        print(f"📁 Uploading {local_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE
        )
        print(f"   ✅ {repo_path} uploaded\n")
    except Exception as e:
        print(f"   ❌ Error uploading {repo_path}: {e}\n")

# Upload audio files directory
print("🎵 Uploading audio files...\n")
audio_dir = Path("data/raw/audio")
if audio_dir.exists():
    wav_files = sorted(list(audio_dir.glob("**/*.wav")))
    print(f"   Found {len(wav_files)} audio files\n")
    for i, wav_file in enumerate(wav_files, 1):
        try:
            relative_path = wav_file.relative_to("data/raw")
            print(f"   [{i}/{len(wav_files)}] Uploading {relative_path}...")
            api.upload_file(
                path_or_fileobj=str(wav_file),
                path_in_repo=f"audio/{relative_path}",
                repo_id=REPO_ID,
                repo_type=REPO_TYPE
            )
        except Exception as e:
            print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ ALL FILES UPLOADED SUCCESSFULLY!")
print("="*60)
print(f"\n📊 View your dataset at:")
print(f"   https://huggingface.co/datasets/{REPO_ID}")
print("\n")
