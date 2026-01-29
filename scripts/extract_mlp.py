import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(PROJECT_ROOT, "weights", "model.safetensors")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.pack_weights import pack_tensor_4bit

BLOCK_SIZE = 32  # industry standard

def extract_mlp(layer: int):
    print(f"Extracting MLP weights for layer {layer}...")
    TENSOR_NAMES = [
        f"model.layers.{layer}.mlp.down_proj.weight",
        f"model.layers.{layer}.mlp.up_proj.weight",
        f"model.layers.{layer}.mlp.gate_proj.weight",
    ]
    for name in TENSOR_NAMES:
        pack_tensor_4bit(WEIGHT_PATH, name)
    print(f"Extracted MLP weights for layer {layer}")