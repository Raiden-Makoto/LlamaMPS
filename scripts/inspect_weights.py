import json
import struct
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(PROJECT_ROOT, "weights", "model.safetensors")

def inspect_weights(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        # 1. The first 8 bytes are a Little-Endian unsigned 64-bit integer
        # This defines the length of the JSON header
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        # 2. Read the next 'header_len' bytes as UTF-8 JSON
        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode("utf-8"))
        # 3. Pull Metadata (if any)
        metadata = header.pop("__metadata__", {})
        print(f"--- Safetensors Inspection: {os.path.basename(path)} ---")
        print(f"Format: {metadata.get('format', 'Unknown')}")
        print(f"Header Size: {header_len / 1024:.2f} KB")
        print("-" * 50)
        # 4. Iterate through tensors and print shapes/offsets
        # We'll look at the first 10 for a quick peek
        print(f"{'Tensor Name':<45} | {'Shape':<15} | {'Dtype':<6}")
        print("-" * 75)
        for i, (name, info) in enumerate(header.items()):
            shape = str(info['shape'])
            dtype = info['dtype']
            print(f"{name[:44]:<45} | {shape:<15} | {dtype:<6}")
            if i >= 15: # Just show the first few layers + the last one
                print("...")
                break
        
        # 5. The Sigma Detail: Calculate data start point
        # Tensor data starts immediately after the 8-byte length + the header JSON
        data_start = 8 + header_len
        print("-" * 75)
        print(f"Binary Data starts at byte: {data_start}")
        print(f"Use this as your base offset for Metal MTLBuffer mapping.")

if __name__ == "__main__":
    inspect_weights(WEIGHT_PATH)