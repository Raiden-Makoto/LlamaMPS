import os
import sys
import torch  # type: ignore
from safetensors.torch import load_file  # type: ignore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "model.safetensors")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "weights", "global")

# Ensure the repo root (which contains the 'utils' package) is on sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.pack_weights import pack_tensor_4bit


def extract_tied_weights(model_path: str, output_dir: str) -> None:
    state_dict = load_file(model_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Input Embeddings (Lookup - Keep as FP16)
    # This is the matrix used for both input and output
    shared_weight = state_dict["model.embed_tokens.weight"].to(torch.float16).numpy()
    shared_weight.tofile(os.path.join(output_dir, "embed_tokens.bin"))

    # 2. Final Layer Norm (FP16)
    final_norm = state_dict["model.norm.weight"].to(torch.float16).numpy()
    final_norm.tofile(os.path.join(output_dir, "final_norm.bin"))

    # 3. Output Head (4-bit Quantization)
    # Even though it's the same data, we pack it for the GEMV kernel
    print("Packing LM Head (shared weights)...")
    pack_tensor_4bit(model_path, "model.embed_tokens.weight")
    print(f"Global weights extracted to {output_dir}")


def extract_globals(model_path: str, output_dir: str) -> None:
    """
    Backwards-compatible entry point that just uses the tied-weight extractor.
    """
    extract_tied_weights(model_path, output_dir)


if __name__ == "__main__":
    extract_globals(MODEL_PATH, OUTPUT_DIR)