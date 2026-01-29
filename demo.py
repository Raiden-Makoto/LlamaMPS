import os
import sys
import numpy as np

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
    logits = engine.run_inference(input_vec)
    next_token_id = int(np.argmax(logits))
    print(f"Input token IDs: {token_ids}")
    print(f"Logits shape: {logits.shape}")
    print(f"Next token ID: {next_token_id}")


if __name__ == "__main__":
    main()
