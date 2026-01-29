import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(PROJECT_ROOT, "weights", "model.safetensors")

# Ensure the repo root (which contains the 'utils' package) is on sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.pack_weights import pack_tensor_4bit

BLOCK_SIZE = 32  # industry standard

def extract_attn(layer: int):
    print(f"Extracting Attention weights for layer {layer}...")
    TENSOR_NAMES = [
        f"model.layers.{layer}.self_attn.q_proj.weight",
        f"model.layers.{layer}.self_attn.k_proj.weight",
        f"model.layers.{layer}.self_attn.v_proj.weight",
        f"model.layers.{layer}.self_attn.o_proj.weight",
        f"model.layers.{layer}.self_attn.c_proj.weight",
    ]
    for name in TENSOR_NAMES:
        pack_tensor_4bit(WEIGHT_PATH, name)
    print(f"Extracted Attention weights for layer {layer}")