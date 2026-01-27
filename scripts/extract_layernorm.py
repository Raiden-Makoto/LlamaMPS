import torch # type: ignore
from safetensors.torch import load_file # type: ignore
import numpy as np # type: ignore

import os

# Path to your downloaded Qwen model
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "model.safetensors")

def extract_layernorm():
    print("Extracting Layer 0 RMSNorm weights...")
    state_dict = load_file(MODEL_PATH)
    
    # Qwen-2.5-1.5B identifier for the first layer's norm
    norm_key = "model.layers.0.input_layernorm.weight"
    
    if norm_key in state_dict:
        # Convert BF16 -> FP16 for the Metal GPU
        norm_weights = state_dict[norm_key].to(torch.float16).numpy()
        out_dir = os.path.join(PROJECT_ROOT, "weights")
        os.makedirs(out_dir, exist_ok=True)
        norm_path = os.path.join(out_dir, "layer0_norm.bin")
        norm_weights.tofile(norm_path)
        print(f"Saved to {norm_path} (Size: {norm_weights.shape})")
    else:
        print(f"Key not found. Available keys: {list(state_dict.keys())[:5]}")

if __name__ == "__main__":
    extract_layernorm()