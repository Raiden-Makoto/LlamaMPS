import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Silence `huggingface/tokenizers` fork/parallelism warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from transformers import AutoTokenizer

from runners.engine import QwenEngine

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
# Prefer the local tokenizer so token-id order matches `weights/tokenizer.json`.
# Fallback to HF if the local folder isn't a complete tokenizer package yet.
try:
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

def _get_stop_token_ids(tok):
    # Prefer tokenizer-defined EOS. Also include Qwen chat template tokens.
    stop_ids = []
    if tok.eos_token_id is not None:
        stop_ids.append(int(tok.eos_token_id))

    for s in ("<|endoftext|>", "<|im_end|>"):
        try:
            tid = tok.convert_tokens_to_ids(s)
        except Exception:
            tid = None
        if tid is not None and tid != tok.unk_token_id:
            stop_ids.append(int(tid))

    # De-duplicate without using dict/set (keep first occurrence).
    uniq = []
    for x in stop_ids:
        seen = False
        for y in uniq:
            if y == x:
                seen = True
                break
        if not seen:
            uniq.append(x)
    return uniq


def main():
    engine = QwenEngine(WEIGHTS_DIR)
    prompt = "The capital of Canada is"

    # For instruct models, use the chat template so the model sees the format it was tuned on.
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    else:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
    # Prefer engine-computed tokenizer max-id+1 (derived from weights/tokenizer.json).
    valid_vocab_size = getattr(engine, "lm_head_rows", max(tokenizer.get_vocab().values()) + 1)
    embed_matrix = np.fromfile(
        os.path.join(WEIGHTS_DIR, "global", "embed_tokens.bin"),
        dtype=np.float16,
    ).reshape(-1, 1536)

    print(f"Prompt: {prompt}", end="", flush=True)

    # Prefill KV-cache with the prompt tokens (so the first generated token is conditioned on the prompt).
    last_logits = None
    for tid in tokens:
        last_logits = engine.run_inference(embed_matrix[tid])

    # Generate 10 tokens
    stop_token_ids = _get_stop_token_ids(tokenizer)
    for step in range(10):
        # The logits from the previous forward pass correspond to the next-token distribution.
        logits = last_logits
        # Cast to float32 before argmax to reduce 4-bit tie-breaking artifacts.
        logits = np.asarray(logits, dtype=np.float32)
        next_id = engine.predict_next_token(logits, valid_vocab_size=valid_vocab_size)

        if next_id in stop_token_ids:
            break

        tokens.append(next_id)
        word = tokenizer.decode([next_id])
        print(word, end="", flush=True)
        last_logits = engine.run_inference(embed_matrix[next_id])

if __name__ == "__main__":
    main()
