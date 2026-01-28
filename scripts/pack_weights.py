import os
import torch # type: ignore
import numpy as np # type: ignore
from safetensors.torch import load_file # type: ignore
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(PROJECT_ROOT, "weights", "model.safetensors")

BLOCK_SIZE = 32  # industry standard

TENSOR_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.c_proj.weight",
]


def pack_tensor_4bit(path: str, tensor_name: str):
    print(f"Loading {tensor_name}...")
    try:
        weights = load_file(path)[tensor_name].to(torch.float32)
    except KeyError:
        warnings.warn(f"Tensor not found: {tensor_name}. Skipping...")
        return
    flat = weights.flatten()
    pad_len = (BLOCK_SIZE - (len(flat) % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_len: flat = torch.cat([flat, torch.zeros(pad_len)])
    # Reshape to blocks
    blocks = flat.view(-1, BLOCK_SIZE)
    # Calculate Scales (Symmetric: Scale = max_abs / 7)
    # Range of INT4 is [-8, 7]. We map max_abs to 7 to preserve zero exactly.
    scales = torch.max(torch.abs(blocks), dim=1).values / 7.0
    scales[scales == 0] = 1.0 # Protect against dead weights (prevent division by zero)
    # Quantize to INT4
    # Formula: round(weight / scale) clamped to [-8, 7]
    q_blocks = torch.round(blocks / scales.unsqueeze(1)).clamp(-8, 7).to(torch.int8)
    # Bit-Packing (Manual Surgery)
    # Shift to unsigned [0, 15] range: -8 becomes 0, 7 becomes 15.
    u_weights = (q_blocks + 8).to(torch.uint8).flatten()
    # Pack two 4-bit weights into one 8-bit byte (Weight_Low | (Weight_High << 4))
    low_half = u_weights[0::2]
    high_half = u_weights[1::2]
    packed = (low_half | (high_half << 4)).numpy().astype(np.uint8)
    # Save Artifacts (basename from tensor, e.g. q_proj -> q_proj_4bit.bin)
    base = tensor_name.replace("model.layers.0.self_attn.", "").replace(".weight", "")
    out_dir = os.path.join(PROJECT_ROOT, "weights")
    os.makedirs(out_dir, exist_ok=True)
    packed_path = os.path.join(out_dir, f"{base}_4bit.bin")
    scales_path = os.path.join(out_dir, f"{base}_scales.bin")
    packed.tofile(packed_path)
    scales.numpy().astype(np.float16).tofile(scales_path)
    print(f"Saved {tensor_name} to {packed_path}")
    print(f"Saved scales to {scales_path}")


if __name__ == "__main__":
    for name in TENSOR_NAMES:
        pack_tensor_4bit(WEIGHT_PATH, name)