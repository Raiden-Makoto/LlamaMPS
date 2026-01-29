import json
import os
import sys

import numpy as np


HIDDEN_DIM = 1536


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _tokenizer_max_id(tokenizer_json_path: str) -> int:
    tok = _load_json(tokenizer_json_path)
    model_vocab = tok.get("model", {}).get("vocab", {})
    ids = []
    if isinstance(model_vocab, dict):
        ids.extend(list(model_vocab.values()))
    added = tok.get("added_tokens", [])
    for t in added:
        if isinstance(t, dict) and "id" in t:
            ids.append(int(t["id"]))
    if not ids:
        return -1
    return int(max(ids))


def _dequant_row_q4_0(packed: np.ndarray, scales: np.ndarray, row: int, k: int) -> np.ndarray:
    """
    Dequantize a single row from the q4_0 layout used by `kernels/quant_matmul.metal`.
    """
    bytes_per_row = k // 2
    scales_per_row = k // 32

    row_packed = packed[row * bytes_per_row : (row + 1) * bytes_per_row]
    row_scales = scales[row * scales_per_row : (row + 1) * scales_per_row].astype(np.float32, copy=False)

    out = np.empty((k,), dtype=np.float32)
    for byte_idx in range(bytes_per_row):
        b = int(row_packed[byte_idx])
        w0 = (b & 0x0F) - 8
        w1 = (b >> 4) - 8
        block_idx = byte_idx // 16  # 16 bytes == 32 weights
        s = float(row_scales[block_idx])
        out[byte_idx * 2] = float(w0) * s
        out[byte_idx * 2 + 1] = float(w1) * s
    return out


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "weights")

    cfg_path = os.path.join(weights_dir, "config.json")
    tok_path = os.path.join(weights_dir, "tokenizer.json")

    if not os.path.exists(cfg_path) or not os.path.exists(tok_path):
        raise FileNotFoundError("Expected `weights/config.json` and `weights/tokenizer.json` to exist.")

    cfg = _load_json(cfg_path)
    vocab_size = int(cfg.get("vocab_size", -1))
    hidden = int(cfg.get("hidden_size", -1))
    layers = int(cfg.get("num_hidden_layers", -1))
    num_q = int(cfg.get("num_attention_heads", -1))
    num_kv = int(cfg.get("num_key_value_heads", -1))
    intermediate = int(cfg.get("intermediate_size", -1))

    max_tok_id = _tokenizer_max_id(tok_path)
    valid_vocab_size = max_tok_id + 1

    print("== Config vs tokenizer ==")
    print("config.vocab_size:", vocab_size)
    print("tokenizer max_id:", max_tok_id)
    print("tokenizer valid_vocab_size:", valid_vocab_size)
    print("config.hidden_size:", hidden)
    print("config.num_hidden_layers:", layers)
    print("config.num_attention_heads:", num_q)
    print("config.num_key_value_heads:", num_kv)
    print("config.intermediate_size:", intermediate)
    print()

    # Basic layout checks (extracted weights).
    global_dir = os.path.join(weights_dir, "global")
    expected = [
        os.path.join(global_dir, "embed_tokens.bin"),
        os.path.join(global_dir, "final_norm.bin"),
        os.path.join(weights_dir, "embed_tokens_4bit.bin"),
        os.path.join(weights_dir, "embed_tokens_scales.bin"),
    ]
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        print("Missing extracted weight artifacts:")
        for p in missing:
            print(" -", os.path.relpath(p, project_root))
        print()
        print("To generate them, run:")
        print("  python scripts/download.py")
        print("  python scripts/extract_global.py")
        print("  python scripts/extract_all.py")
        sys.exit(1)

    if hidden != HIDDEN_DIM:
        print(f"WARNING: verify script expects hidden={HIDDEN_DIM}, but config says hidden={hidden}")

    # Per-layer artifact checks (sizes match the q4_0 GEMV kernel assumptions).
    print("== Per-layer artifacts ==")
    bytes_per_row_1536 = hidden // 2
    scales_per_row_1536 = hidden // 32

    # Qwen2.5-1.5B specific: K/V projection output dim is 256.
    k_dim = int(cfg.get("num_key_value_heads", 2)) * int(cfg.get("hidden_size", hidden)) // int(cfg.get("num_attention_heads", 12))
    # For this repo we expect 256; keep the math so config drives it.
    print("derived k_dim:", k_dim)

    def _check_size(path: str, expected_bytes: int) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        actual = os.path.getsize(path)
        if actual != expected_bytes:
            raise ValueError(f"Bad size: {path} expected {expected_bytes} bytes, got {actual} bytes")

    for layer in range(layers):
        layer_dir = os.path.join(weights_dir, f"layer{layer}")
        if not os.path.isdir(layer_dir):
            raise FileNotFoundError(f"Missing layer directory: {layer_dir}")

        _check_size(os.path.join(layer_dir, f"layer{layer}_norm.bin"), hidden * 2)
        _check_size(os.path.join(layer_dir, f"layer{layer}_post_attn_norm.bin"), hidden * 2)

        # QKV + O (K=hidden)
        for name, rows in [("q_proj", hidden), ("k_proj", k_dim), ("v_proj", k_dim), ("o_proj", hidden)]:
            _check_size(os.path.join(layer_dir, f"{name}_4bit.bin"), rows * bytes_per_row_1536)
            _check_size(os.path.join(layer_dir, f"{name}_scales.bin"), rows * scales_per_row_1536 * 2)

        # MLP gate/up: [intermediate, hidden]
        for name in ("gate_proj", "up_proj"):
            _check_size(os.path.join(layer_dir, f"{name}_4bit.bin"), intermediate * bytes_per_row_1536)
            _check_size(os.path.join(layer_dir, f"{name}_scales.bin"), intermediate * scales_per_row_1536 * 2)

        # MLP down: [hidden, intermediate]  (K=intermediate)
        _check_size(os.path.join(layer_dir, "down_proj_4bit.bin"), hidden * (intermediate // 2))
        _check_size(os.path.join(layer_dir, "down_proj_scales.bin"), hidden * (intermediate // 32) * 2)

    print(f"OK: found {layers} layer folders with expected file sizes")
    print()

    # Spot-check tied embedding order by comparing FP16 rows to q4_0 dequant rows.
    k = hidden
    fp16_embed = np.memmap(os.path.join(global_dir, "embed_tokens.bin"), dtype=np.float16, mode="r")
    packed = np.memmap(os.path.join(weights_dir, "embed_tokens_4bit.bin"), dtype=np.uint8, mode="r")
    scales = np.memmap(os.path.join(weights_dir, "embed_tokens_scales.bin"), dtype=np.float16, mode="r")

    rows_fp16 = fp16_embed.size // k
    print("== Embedding matrix ==")
    print("rows(fp16):", rows_fp16)
    print("rows(config.vocab_size):", vocab_size)
    print("rows(tokenizer valid_vocab_size):", valid_vocab_size)
    print()

    sample_ids = [0, 1, 2, 10, 100, 1000, min(valid_vocab_size - 1, rows_fp16 - 1)]
    sample_ids = [i for i in sample_ids if 0 <= i < rows_fp16]

    print("== Tied embedding spot-check (row order) ==")
    for tid in sample_ids:
        a = fp16_embed[tid * k : (tid + 1) * k].astype(np.float32, copy=False)
        b = _dequant_row_q4_0(packed, scales, tid, k)
        # cosine similarity
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        cos = float(np.dot(a, b) / denom)
        mae = float(np.mean(np.abs(a - b)))
        print(f"id={tid:6d}  cos={cos: .6f}  mae={mae: .6f}")


if __name__ == "__main__":
    main()

