import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(PROJECT_ROOT, "weights", "model.safetensors")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.pack_weights import pack_tensor_4bit

BLOCK_SIZE = 32  # industry standard

TENSOR_NAMES = [
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
]

if __name__ == "__main__":
    for name in TENSOR_NAMES:
        pack_tensor_4bit(WEIGHT_PATH, name)