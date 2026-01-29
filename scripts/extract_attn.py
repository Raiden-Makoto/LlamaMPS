import os
import sys
import torch  # type: ignore
from safetensors.torch import load_file  # type: ignore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(PROJECT_ROOT, "weights", "model.safetensors")

# Ensure the repo root (which contains the 'utils' package) is on sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.pack_weights import pack_tensor_4bit

BLOCK_SIZE = 32  # industry standard

def extract_attn(layer: int):
    print(f"Extracting Attention weights for layer {layer}...")
    weight_names = [
        f"model.layers.{layer}.self_attn.q_proj.weight",
        f"model.layers.{layer}.self_attn.k_proj.weight",
        f"model.layers.{layer}.self_attn.v_proj.weight",
        f"model.layers.{layer}.self_attn.o_proj.weight",
    ]
    for name in weight_names:
        pack_tensor_4bit(WEIGHT_PATH, name)

    # Qwen2/Qwen2.5 uses bias on q/k/v projections; extract them to FP16.
    state_dict = load_file(WEIGHT_PATH)
    out_dir = os.path.join(PROJECT_ROOT, "weights")
    os.makedirs(out_dir, exist_ok=True)
    for proj in ("q_proj", "k_proj", "v_proj"):
        k = f"model.layers.{layer}.self_attn.{proj}.bias"
        if k in state_dict:
            b = state_dict[k].to(torch.float16).numpy()
            b.tofile(os.path.join(out_dir, f"{proj}_bias.bin"))
    print(f"Extracted Attention weights for layer {layer}")