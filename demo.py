import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runners.engine import QwenEngine
from utils.tokenizer import get_input_vector

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")


def main():
    engine = QwenEngine(WEIGHTS_DIR)
    prompt = "The capital of Canada is"
    input_vec, token_ids = get_input_vector(prompt)
    result = engine.run_inference(input_vec)
    print(f"Token IDs: {token_ids}")
    print(f"Output shape: {result.shape}")
    print(f"Output (first 5): {result[:5]}")


if __name__ == "__main__":
    main()
