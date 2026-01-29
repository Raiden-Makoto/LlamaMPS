from transformers import AutoTokenizer
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EMBED_PATH = os.path.join(PROJECT_ROOT, "weights", "global", "embed_tokens.bin")

# 1. Load the official Qwen 2.5 Tokenizer
# This handles the complex BPE merges and special tokens
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")


def get_input_vector(prompt, embed_path=None):
    if embed_path is None:
        embed_path = DEFAULT_EMBED_PATH
    # 2. Convert text to IDs
    # e.g., "Hello" -> [14920]
    tokens = tokenizer.encode(prompt, add_special_tokens=True)

    # 3. Embedding Lookup (The "Input" to Layer 0)
    # Load the 151,936 x 1536 matrix we saved earlier
    embedding_matrix = np.fromfile(embed_path, dtype=np.float16).reshape(-1, 1536)
    
    # Pull the vector for the FIRST token to start our test
    first_token_id = tokens[0]
    input_vector = embedding_matrix[first_token_id]
    
    return input_vector, tokens

# Test it
if __name__ == "__main__":
    prompt = "The capital of Canada is"
    vec, ids = get_input_vector(prompt)
    print(f"Starting Vector Shape: {vec.shape}") # Should be (1536,)
    print(f"Token IDs: {ids}")