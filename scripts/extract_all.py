import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_ROOT = os.path.join(PROJECT_ROOT, "weights")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_attn import extract_attn
from scripts.extract_mlp import extract_mlp
from scripts.extract_layernorm import extract_layernorm
from scripts.extract_postnorm import extract_post_norm


NUM_LAYERS = 28


def copy_layer_weights(layer: int) -> None:
    """Move the just-extracted flat weight files into a per-layer subfolder."""
    layer_dir = os.path.join(WEIGHTS_ROOT, f"layer{layer}")
    os.makedirs(layer_dir, exist_ok=True)

    # Files produced by attention + MLP packers (layer-agnostic names)
    common_bins = [
        # Attention QKVOC
        "q_proj_4bit.bin", "q_proj_scales.bin",
        "q_proj_bias.bin",
        "k_proj_4bit.bin", "k_proj_scales.bin",
        "k_proj_bias.bin",
        "v_proj_4bit.bin", "v_proj_scales.bin",
        "v_proj_bias.bin",
        "o_proj_4bit.bin", "o_proj_scales.bin",
        # MLP gate / up / down
        "gate_proj_4bit.bin", "gate_proj_scales.bin",
        "up_proj_4bit.bin", "up_proj_scales.bin",
        "down_proj_4bit.bin", "down_proj_scales.bin",
    ]

    # Norms already encode the layer index in their filenames
    norm_bins = [
        f"layer{layer}_norm.bin",
        f"layer{layer}_post_attn_norm.bin",
    ]

    for name in common_bins + norm_bins:
        src = os.path.join(WEIGHTS_ROOT, name)
        if os.path.exists(src):
            dst = os.path.join(layer_dir, name)
            shutil.move(src, dst)


if __name__ == "__main__":
    for layer in range(NUM_LAYERS):
        # 1. Extract flat weights for this layer into the shared weights folder
        extract_layernorm(layer)
        extract_attn(layer)
        extract_post_norm(layer)
        extract_mlp(layer)
        # 2. Copy each weight into the layer's subfolder
        copy_layer_weights(layer)