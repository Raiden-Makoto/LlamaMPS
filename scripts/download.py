import os
from huggingface_hub import hf_hub_download # type: ignore

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_DIR = "./weights"

FILES = [
    "model.safetensors",
    "tokenizer.json",
    "config.json",
]

def download_model():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for fname in FILES:
        print(f"Downloading {fname}...")
        try:
            pth = hf_hub_download(
                repo_id=MODEL_ID,
                filename=fname,
                local_dir=TARGET_DIR,
                local_dir_use_symlinks=False, # keep it simple
            )
            print(f"Downloaded to {pth}")
        except Exception as e:
            print(f"Error downloading {fname}: {e}. You may need to run `huggingface-cli login` to authenticate.")

if __name__ == "__main__":
    download_model()