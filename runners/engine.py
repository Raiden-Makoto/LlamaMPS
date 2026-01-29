import Metal
import numpy as np
import json
import os
import time

# --- Constants for Qwen 2.5-1.5B ---
HIDDEN_DIM = 1536
MLP_INTERMEDIATE = 8960
VOCAB_SIZE = 151936
NUM_LAYERS = 28
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
K_DIM = 256
MAX_SEQ_LEN = 1024
GEMV_THREADS = 768

class QwenEngine:
    def __init__(self, weights_dir):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.cmd_queue = self.device.newCommandQueue()
        self.weights_dir = weights_dir
        self.current_pos = 0

        self.lm_head_rows = self._infer_tokenizer_vocab_size()
        self._validate_weights_dir()

        # Load Weights
        self.layer_weights = self._preload_weights()
        global_dir = os.path.join(self.weights_dir, "global")
        self.buf_final_norm_w = self._create_buffer(np.fromfile(os.path.join(global_dir, "final_norm.bin"), dtype=np.float16))
        self.buf_lm_head_packed = self._create_buffer(np.fromfile(os.path.join(self.weights_dir, "embed_tokens_4bit.bin"), dtype=np.uint8))
        self.buf_lm_head_scales = self._create_buffer(np.fromfile(os.path.join(self.weights_dir, "embed_tokens_scales.bin"), dtype=np.float16))

        # Compiling (Paths assumed relative to project root)
        self.kernels = {
            "norm": self._get_pso("rms_norm.metal", "rms_norm_q4"),
            "gemv": self._get_pso("quant_matmul.metal", "gemv_q4_0"),
            "bias_add": self._get_pso("bias_add.metal", "bias_add"),
            "rope": self._get_pso("rope.metal", "apply_rope_q4"),
            "attn": self._get_pso("attention.metal", "attention_scores"),
            "softmax": self._get_pso("softmax.metal", "softmax"),
            "attn_sum": self._get_pso("attn_sum.metal", "attn_weighted_sum"),
            "resid": self._get_pso("residual.metal", "residual_add"),
            "swiglu": self._get_pso("swiglu.metal", "swiglu"),
        }

        self._create_buffer_pool()
        
        # State Buffers
        self.buf_layer_idx = self.device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)
        self.buf_pos = self.device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)
        self.buf_const_1536 = self._create_buffer(np.array([1536, 768 // 32, 768], dtype=np.uint32))
        self.buf_const_8960 = self._create_buffer(np.array([8960, 768 // 32, 768], dtype=np.uint32))

    def _validate_weights_dir(self) -> None:
        def _req(path: str) -> None:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing weight artifact: {path}\n"
                    f"Expected you to run: python scripts/download.py && "
                    f"python scripts/extract_global.py && python scripts/extract_all.py"
                )

        def _req_size(path: str, expected_bytes: int) -> None:
            _req(path)
            actual = os.path.getsize(path)
            if actual != expected_bytes:
                raise ValueError(
                    f"Bad weight file size for {path}\n"
                    f"Expected {expected_bytes} bytes, got {actual} bytes"
                )

        def _req_min_size(path: str, min_bytes: int) -> None:
            _req(path)
            actual = os.path.getsize(path)
            if actual < min_bytes:
                raise ValueError(
                    f"Bad weight file size for {path}\n"
                    f"Expected at least {min_bytes} bytes, got {actual} bytes"
                )

        # Validate config matches our hardcoded constants (catches model-size mismatches).
        cfg_path = os.path.join(self.weights_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            expected = [
                ("hidden_size", HIDDEN_DIM),
                ("intermediate_size", MLP_INTERMEDIATE),
                ("vocab_size", VOCAB_SIZE),
                ("num_hidden_layers", NUM_LAYERS),
                ("num_attention_heads", NUM_Q_HEADS),
                ("num_key_value_heads", NUM_KV_HEADS),
            ]
            for k, v in expected:
                if k in cfg and int(cfg[k]) != int(v):
                    raise ValueError(f"weights/config.json mismatch: {k}={cfg[k]} but engine expects {v}")

        # Global artifacts (tied embeddings + final norm).
        global_dir = os.path.join(self.weights_dir, "global")
        _req_size(os.path.join(global_dir, "final_norm.bin"), HIDDEN_DIM * 2)

        # Tied LM head (packed view of embed_tokens.weight)
        bytes_per_row = HIDDEN_DIM // 2
        scales_per_row = HIDDEN_DIM // 32
        # These files may include extra reserved rows beyond tokenizer.json; we only require
        # enough rows to cover the tokenizer's max-id+1 (self.lm_head_rows).
        _req_min_size(os.path.join(self.weights_dir, "embed_tokens_4bit.bin"), self.lm_head_rows * bytes_per_row)
        _req_min_size(os.path.join(self.weights_dir, "embed_tokens_scales.bin"), self.lm_head_rows * scales_per_row * 2)

        # Per-layer artifacts.
        for i in range(NUM_LAYERS):
            layer_dir = os.path.join(self.weights_dir, f"layer{i}")
            # Norms
            _req_size(os.path.join(layer_dir, f"layer{i}_norm.bin"), HIDDEN_DIM * 2)
            _req_size(os.path.join(layer_dir, f"layer{i}_post_attn_norm.bin"), HIDDEN_DIM * 2)

            # QKV + O (K=1536)
            for name, rows in [("q_proj", HIDDEN_DIM), ("k_proj", K_DIM), ("v_proj", K_DIM), ("o_proj", HIDDEN_DIM)]:
                _req_size(os.path.join(layer_dir, f"{name}_4bit.bin"), rows * (HIDDEN_DIM // 2))
                _req_size(os.path.join(layer_dir, f"{name}_scales.bin"), rows * (HIDDEN_DIM // 32) * 2)
            # Qwen2/Qwen2.5 uses q/k/v biases.
            _req_size(os.path.join(layer_dir, "q_proj_bias.bin"), HIDDEN_DIM * 2)
            _req_size(os.path.join(layer_dir, "k_proj_bias.bin"), K_DIM * 2)
            _req_size(os.path.join(layer_dir, "v_proj_bias.bin"), K_DIM * 2)

            # MLP gate/up (K=1536) and down (K=8960)
            for name in ("gate_proj", "up_proj"):
                _req_size(os.path.join(layer_dir, f"{name}_4bit.bin"), MLP_INTERMEDIATE * (HIDDEN_DIM // 2))
                _req_size(os.path.join(layer_dir, f"{name}_scales.bin"), MLP_INTERMEDIATE * (HIDDEN_DIM // 32) * 2)

            _req_size(os.path.join(layer_dir, "down_proj_4bit.bin"), HIDDEN_DIM * (MLP_INTERMEDIATE // 2))
            _req_size(os.path.join(layer_dir, "down_proj_scales.bin"), HIDDEN_DIM * (MLP_INTERMEDIATE // 32) * 2)

    def _infer_tokenizer_vocab_size(self) -> int:
        """
        Returns (max_token_id + 1) from `weights/tokenizer.json`.
        This is the *actual* set of token ids we should score in the LM head.
        """
        tok_path = os.path.join(self.weights_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            return VOCAB_SIZE

        with open(tok_path, "r") as f:
            tok = json.load(f)

        ids = []
        model_vocab = tok.get("model", {}).get("vocab", {})
        if isinstance(model_vocab, dict):
            # values are token ids
            ids.extend(list(model_vocab.values()))

        added = tok.get("added_tokens", [])
        for t in added:
            if isinstance(t, dict) and "id" in t:
                ids.append(int(t["id"]))

        if not ids:
            return VOCAB_SIZE

        max_id = int(max(ids))
        return max_id + 1

    def _get_pso(self, filename, func_name):
        path = os.path.join("kernels", filename)
        with open(path, "r") as f: source = f.read()
        lib, _ = self.device.newLibraryWithSource_options_error_(source, None, None)
        func = lib.newFunctionWithName_(func_name)
        pso, _ = self.device.newComputePipelineStateWithFunction_error_(func, None)
        return pso

    def _create_buffer(self, arr):
        return self.device.newBufferWithBytes_length_options_(arr.tobytes(), len(arr.tobytes()), Metal.MTLResourceStorageModeShared)

    def _create_buffer_pool(self):
        self.buf_residual = self.device.newBufferWithLength_options_(HIDDEN_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_x = self.device.newBufferWithLength_options_(HIDDEN_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_normed = self.device.newBufferWithLength_options_(HIDDEN_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_q = self.device.newBufferWithLength_options_(HIDDEN_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_k = self.device.newBufferWithLength_options_(K_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_v = self.device.newBufferWithLength_options_(K_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_scores = self.device.newBufferWithLength_options_(NUM_Q_HEADS * MAX_SEQ_LEN * 4, Metal.MTLResourceStorageModeShared)
        self.buf_attn_out = self.device.newBufferWithLength_options_(HIDDEN_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_logits = self.device.newBufferWithLength_options_(VOCAB_SIZE * 2, Metal.MTLResourceStorageModeShared)
        self.buf_gate_out = self.device.newBufferWithLength_options_(MLP_INTERMEDIATE * 2, Metal.MTLResourceStorageModeShared)
        self.buf_up_out = self.device.newBufferWithLength_options_(MLP_INTERMEDIATE * 2, Metal.MTLResourceStorageModeShared)
        self.buf_swiglu_out = self.device.newBufferWithLength_options_(MLP_INTERMEDIATE * 2, Metal.MTLResourceStorageModeShared)
        self.buf_mlp_out = self.device.newBufferWithLength_options_(HIDDEN_DIM * 2, Metal.MTLResourceStorageModeShared)
        
        # Contiguous KV Cache
        self.buf_k_cache = self.device.newBufferWithLength_options_(NUM_LAYERS * MAX_SEQ_LEN * K_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_v_cache = self.device.newBufferWithLength_options_(NUM_LAYERS * MAX_SEQ_LEN * K_DIM * 2, Metal.MTLResourceStorageModeShared)

        # Temporary Layer Weight Buffers
        self.buf_w_norm = self.device.newBufferWithLength_options_(HIDDEN_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_w_packed = self.device.newBufferWithLength_options_(MLP_INTERMEDIATE * (HIDDEN_DIM // 2), Metal.MTLResourceStorageModeShared)
        self.buf_w_scales = self.device.newBufferWithLength_options_(MLP_INTERMEDIATE * (HIDDEN_DIM // 32) * 2, Metal.MTLResourceStorageModeShared)
        self.buf_bias = self.device.newBufferWithLength_options_(HIDDEN_DIM * 2, Metal.MTLResourceStorageModeShared)

    def _preload_weights(self):
        layers = []
        for i in range(NUM_LAYERS):
            d = os.path.join(self.weights_dir, f"layer{i}")
            layers.append({
                "in_n": np.fromfile(os.path.join(d, f"layer{i}_norm.bin"), dtype=np.float16),
                "po_n": np.fromfile(os.path.join(d, f"layer{i}_post_attn_norm.bin"), dtype=np.float16),
                "qb": np.fromfile(os.path.join(d, "q_proj_bias.bin"), dtype=np.float16),
                "kb": np.fromfile(os.path.join(d, "k_proj_bias.bin"), dtype=np.float16),
                "vb": np.fromfile(os.path.join(d, "v_proj_bias.bin"), dtype=np.float16),
                "q": np.fromfile(os.path.join(d, "q_proj_4bit.bin"), dtype=np.uint8), "qs": np.fromfile(os.path.join(d, "q_proj_scales.bin"), dtype=np.float16),
                "k": np.fromfile(os.path.join(d, "k_proj_4bit.bin"), dtype=np.uint8), "ks": np.fromfile(os.path.join(d, "k_proj_scales.bin"), dtype=np.float16),
                "v": np.fromfile(os.path.join(d, "v_proj_4bit.bin"), dtype=np.uint8), "vs": np.fromfile(os.path.join(d, "v_proj_scales.bin"), dtype=np.float16),
                "o": np.fromfile(os.path.join(d, "o_proj_4bit.bin"), dtype=np.uint8), "os": np.fromfile(os.path.join(d, "o_proj_scales.bin"), dtype=np.float16),
                "gt": np.fromfile(os.path.join(d, "gate_proj_4bit.bin"), dtype=np.uint8), "gts": np.fromfile(os.path.join(d, "gate_proj_scales.bin"), dtype=np.float16),
                "up": np.fromfile(os.path.join(d, "up_proj_4bit.bin"), dtype=np.uint8), "ups": np.fromfile(os.path.join(d, "up_proj_scales.bin"), dtype=np.float16),
                "dn": np.fromfile(os.path.join(d, "down_proj_4bit.bin"), dtype=np.uint8), "dns": np.fromfile(os.path.join(d, "down_proj_scales.bin"), dtype=np.float16),
            })
        return layers

    def _copy(self, buf, data):
        np.copyto(np.frombuffer(buf.contents().as_buffer(len(data.tobytes())), dtype=np.uint8), data.view(np.uint8))

    def run_inference(self, input_vector: np.ndarray) -> np.ndarray:
        # Clear logits to avoid carry-over from previous tokens.
        # (This is cheap and helps diagnose partial-write bugs in the GEMV kernel.)
        np.copyto(
            np.frombuffer(
                self.buf_logits.contents().as_buffer(VOCAB_SIZE * 2),
                dtype=np.uint16,
            ),
            0,
        )
        self._copy(self.buf_residual, input_vector.astype(np.float16))
        self._copy(self.buf_pos, np.array([self.current_pos], dtype=np.uint32))

        for i in range(NUM_LAYERS):
            self._execute_layer(i)

        cmd = self.cmd_queue.commandBuffer(); enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.kernels["norm"]); enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 0)
        enc.setBuffer_offset_atIndex_(self.buf_final_norm_w, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_x, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(HIDDEN_DIM // 2, 1, 1), Metal.MTLSize(HIDDEN_DIM // 2, 1, 1))

        enc.setComputePipelineState_(self.kernels["gemv"])
        enc.setBuffer_offset_atIndex_(self.buf_lm_head_packed, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_lm_head_scales, 0, 1)
        enc.setBuffer_offset_atIndex_(self.buf_x, 0, 2); enc.setBuffer_offset_atIndex_(self.buf_logits, 0, 3)
        enc.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        # Strictly limit LM head rows to tokenizer max-id+1 to avoid "dead-zone" logits.
        enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(self.lm_head_rows, 1, 1), Metal.MTLSize(768, 1, 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        logits = np.frombuffer(self.buf_logits.contents().as_buffer(VOCAB_SIZE * 2), dtype=np.float16)
        self.current_pos += 1
        return logits

    def predict_next_token(self, logits: np.ndarray, valid_vocab_size=None) -> int:
        # IMPORTANT: cast to float32 *before* argmax to avoid tie-breaking quirks
        # that can show up with 4-bit weights + float16 accumulation.
        logits_f32 = np.asarray(logits, dtype=np.float32)
        if valid_vocab_size is not None:
            logits_f32 = logits_f32[:valid_vocab_size]
        return int(np.argmax(logits_f32))

    def _execute_layer(self, idx):
        w = self.layer_weights[idx]
        self._copy(self.buf_layer_idx, np.array([idx], dtype=np.uint32))
        cmd = self.cmd_queue.commandBuffer(); enc = cmd.computeCommandEncoder()

        # Norm -> QKV
        enc.setComputePipelineState_(self.kernels["norm"]); self._copy(self.buf_w_norm, w["in_n"])
        enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_w_norm, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_normed, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(HIDDEN_DIM // 2, 1, 1), Metal.MTLSize(HIDDEN_DIM // 2, 1, 1))

        enc.setComputePipelineState_(self.kernels["gemv"])
        for pk, sc, bias, out, dim in [
            (w["q"], w["qs"], w["qb"], self.buf_q, HIDDEN_DIM),
            (w["k"], w["ks"], w["kb"], self.buf_k, K_DIM),
            (w["v"], w["vs"], w["vb"], self.buf_v, K_DIM),
        ]:
            self._copy(self.buf_w_packed, pk); self._copy(self.buf_w_scales, sc)
            enc.setBuffer_offset_atIndex_(self.buf_w_packed, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_w_scales, 0, 1)
            enc.setBuffer_offset_atIndex_(self.buf_normed, 0, 2); enc.setBuffer_offset_atIndex_(out, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(dim, 1, 1), Metal.MTLSize(768, 1, 1))

            # Add bias (Qwen2/Qwen2.5 has q/k/v biases)
            self._copy(self.buf_bias, bias)
            enc.setComputePipelineState_(self.kernels["bias_add"])
            enc.setBuffer_offset_atIndex_(out, 0, 0)
            enc.setBuffer_offset_atIndex_(self.buf_bias, 0, 1)
            enc.setBuffer_offset_atIndex_(out, 0, 2)
            enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(dim, 1, 1), Metal.MTLSize(min(dim, 512), 1, 1))
            enc.setComputePipelineState_(self.kernels["gemv"])

        # RoPE -> Attention (Stateful)
        enc.setComputePipelineState_(self.kernels["rope"])
        enc.setBuffer_offset_atIndex_(self.buf_q, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_pos, 0, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(HIDDEN_DIM // 2, 1, 1), Metal.MTLSize(256, 1, 1))
        enc.setBuffer_offset_atIndex_(self.buf_k, 0, 0); enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(K_DIM // 2, 1, 1), Metal.MTLSize(128, 1, 1))

        enc.setComputePipelineState_(self.kernels["attn"])
        enc.setBuffer_offset_atIndex_(self.buf_q, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_k, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_v, 0, 2)
        enc.setBuffer_offset_atIndex_(self.buf_k_cache, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_v_cache, 0, 4); enc.setBuffer_offset_atIndex_(self.buf_scores, 0, 5)
        enc.setBuffer_offset_atIndex_(self.buf_layer_idx, 0, 6); enc.setBuffer_offset_atIndex_(self.buf_pos, 0, 7)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(12, 1, 1), Metal.MTLSize(12, 1, 1))

        enc.setComputePipelineState_(self.kernels["softmax"])
        enc.setBuffer_offset_atIndex_(self.buf_scores, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_pos, 0, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(12, 1, 1), Metal.MTLSize(12, 1, 1))

        enc.setComputePipelineState_(self.kernels["attn_sum"])
        enc.setBuffer_offset_atIndex_(self.buf_scores, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_v_cache, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_attn_out, 0, 2)
        enc.setBuffer_offset_atIndex_(self.buf_layer_idx, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_pos, 0, 4)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(HIDDEN_DIM, 1, 1), Metal.MTLSize(512, 1, 1))

        # MLP
        enc.setComputePipelineState_(self.kernels["gemv"])
        self._copy(self.buf_w_packed, w["o"]); self._copy(self.buf_w_scales, w["os"])
        enc.setBuffer_offset_atIndex_(self.buf_w_packed, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_w_scales, 0, 1)
        enc.setBuffer_offset_atIndex_(self.buf_attn_out, 0, 2); enc.setBuffer_offset_atIndex_(self.buf_x, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(HIDDEN_DIM, 1, 1), Metal.MTLSize(768, 1, 1))

        enc.setComputePipelineState_(self.kernels["resid"])
        enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_x, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(HIDDEN_DIM, 1, 1), Metal.MTLSize(512, 1, 1))

        enc.setComputePipelineState_(self.kernels["norm"]); self._copy(self.buf_w_norm, w["po_n"])
        enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_w_norm, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_normed, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(HIDDEN_DIM // 2, 1, 1), Metal.MTLSize(HIDDEN_DIM // 2, 1, 1))

        enc.setComputePipelineState_(self.kernels["gemv"])
        for pk, sc, out in [(w["gt"], w["gts"], self.buf_gate_out), (w["up"], w["ups"], self.buf_up_out)]:
            self._copy(self.buf_w_packed, pk); self._copy(self.buf_w_scales, sc)
            enc.setBuffer_offset_atIndex_(self.buf_w_packed, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_w_scales, 0, 1)
            enc.setBuffer_offset_atIndex_(self.buf_normed, 0, 2); enc.setBuffer_offset_atIndex_(out, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(MLP_INTERMEDIATE, 1, 1), Metal.MTLSize(768, 1, 1))

        enc.setComputePipelineState_(self.kernels["swiglu"])
        enc.setBuffer_offset_atIndex_(self.buf_gate_out, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_up_out, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_swiglu_out, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(MLP_INTERMEDIATE, 1, 1), Metal.MTLSize(512, 1, 1))

        enc.setComputePipelineState_(self.kernels["gemv"])
        self._copy(self.buf_w_packed, w["dn"]); self._copy(self.buf_w_scales, w["dns"])
        enc.setBuffer_offset_atIndex_(self.buf_w_packed, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_w_scales, 0, 1)
        enc.setBuffer_offset_atIndex_(self.buf_swiglu_out, 0, 2); enc.setBuffer_offset_atIndex_(self.buf_mlp_out, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_const_8960, 0, 4)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(HIDDEN_DIM, 1, 1), Metal.MTLSize(768, 1, 1))

        enc.setComputePipelineState_(self.kernels["resid"])
        enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_mlp_out, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(HIDDEN_DIM, 1, 1), Metal.MTLSize(512, 1, 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()