import os
import torch # type: ignore
from safetensors.torch import load_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "model.safetensors")

def extract_post_norm(layer: int):
    print(f"Extracting Layer {layer} Post-Attention RMSNorm weights...")
    state_dict = load_file(MODEL_PATH)
    
    # Key for the second norm in the block
    norm_key = f"model.layers.{layer}.post_attention_layernorm.weight"
    
    if norm_key in state_dict:
        # Convert to FP16 for Metal
        norm_weights = state_dict[norm_key].to(torch.float16).numpy()
        norm_weights.tofile(os.path.join(PROJECT_ROOT, "weights", f"layer{layer}_post_attn_norm.bin"))
        print(f"Saved to {os.path.join(PROJECT_ROOT, 'weights', f'layer{layer}_post_attn_norm.bin')}")
    else:
        print(f"Key not found: {norm_key}")