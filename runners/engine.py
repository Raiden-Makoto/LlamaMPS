import Metal  # type: ignore
import numpy as np  # type: ignore
import os  # type: ignore
import sys  # type: ignore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIDDEN_DIM = 1536
MLP_INTERMEDIATE = 8960
VOCAB_SIZE = 151936
NUM_LAYERS = 28
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
K_DIM = NUM_KV_HEADS * HEAD_DIM  # 256

# GEMV dispatch: (output_rows, 1, 1) threadgroups
# buffer(4) = [K, simd_groups, threads] where simd_groups = threads/32
GEMV_THREADS = 768
GEMV_SIMD_GROUPS = GEMV_THREADS // 32


def load_bin(path: str, dtype=np.uint8) -> np.ndarray:
    return np.fromfile(path, dtype=dtype)


class QwenEngine:
    def __init__(self, weights_dir: str, device=None):
        self.device = device or Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Metal is not supported on this system.")
        self.cmd_queue = self.device.newCommandQueue()
        self.weights_dir = weights_dir
        self.layers = NUM_LAYERS
        self.hidden_dim = HIDDEN_DIM
        self.mlp_intermediate = MLP_INTERMEDIATE

        self.layer_weights = self._preload_weights()

        self.kernels = {
            "norm": self._get_pso("rms_norm.metal", "rms_norm_q4"),
            "gemv": self._get_pso("quant_matmul.metal", "gemv_q4_0"),
            "rope": self._get_pso("rope.metal", "apply_rope_q4"),
            "attn": self._get_pso("attention.metal", "attention_scores"),
            "softmax": self._get_pso("softmax.metal", "softmax"),
            "attn_sum": self._get_pso("attn_sum.metal", "attn_weighted_sum"),
            "resid": self._get_pso("residual.metal", "residual_add"),
            "swiglu": self._get_pso("swiglu.metal", "swiglu"),
        }

        self._create_buffer_pool()

        global_dir = os.path.join(self.weights_dir, "global")
        final_norm = load_bin(os.path.join(global_dir, "final_norm.bin"), np.float16)
        self.buf_final_norm = self._create_buffer(final_norm)

        self.lm_head_packed = load_bin(os.path.join(self.weights_dir, "embed_tokens_4bit.bin"))
        self.lm_head_scales = load_bin(os.path.join(self.weights_dir, "embed_tokens_scales.bin"), np.float16)
        self.buf_lm_head_packed = self._create_buffer(self.lm_head_packed)
        self.buf_lm_head_scales = self.device.newBufferWithLength_options_(len(self.lm_head_scales) * 2, Metal.MTLResourceStorageModeShared)
        dest = np.frombuffer(self.buf_lm_head_scales.contents().as_buffer(len(self.lm_head_scales) * 2), dtype=np.uint8)
        dest[:] = self.lm_head_scales.view(np.uint8)

        self.buf_const_1536 = self._create_buffer(np.array([1536, GEMV_SIMD_GROUPS, GEMV_THREADS], dtype=np.uint32))
        self.buf_const_8960 = self._create_buffer(np.array([8960, GEMV_SIMD_GROUPS, GEMV_THREADS], dtype=np.uint32))
        self.buf_seq_len = self._create_buffer(np.array([1], dtype=np.uint32))

    def _layer_dir(self, layer: int) -> str:
        return os.path.join(self.weights_dir, f"layer{layer}")

    def _preload_weights(self) -> list:
        weights = []
        for i in range(self.layers):
            layer_dir = self._layer_dir(i)
            w = {
                "input_norm": load_bin(os.path.join(layer_dir, f"layer{i}_norm.bin"), np.float16),
                "post_norm": load_bin(os.path.join(layer_dir, f"layer{i}_post_attn_norm.bin"), np.float16),
                "q": load_bin(os.path.join(layer_dir, "q_proj_4bit.bin")),
                "q_scales": load_bin(os.path.join(layer_dir, "q_proj_scales.bin"), np.float16),
                "k": load_bin(os.path.join(layer_dir, "k_proj_4bit.bin")),
                "k_scales": load_bin(os.path.join(layer_dir, "k_proj_scales.bin"), np.float16),
                "v": load_bin(os.path.join(layer_dir, "v_proj_4bit.bin")),
                "v_scales": load_bin(os.path.join(layer_dir, "v_proj_scales.bin"), np.float16),
                "o": load_bin(os.path.join(layer_dir, "o_proj_4bit.bin")),
                "o_scales": load_bin(os.path.join(layer_dir, "o_proj_scales.bin"), np.float16),
                "gate": load_bin(os.path.join(layer_dir, "gate_proj_4bit.bin")),
                "gate_scales": load_bin(os.path.join(layer_dir, "gate_proj_scales.bin"), np.float16),
                "up": load_bin(os.path.join(layer_dir, "up_proj_4bit.bin")),
                "up_scales": load_bin(os.path.join(layer_dir, "up_proj_scales.bin"), np.float16),
                "down": load_bin(os.path.join(layer_dir, "down_proj_4bit.bin")),
                "down_scales": load_bin(os.path.join(layer_dir, "down_proj_scales.bin"), np.float16),
            }
            weights.append(w)
        return weights

    def _get_pso(self, filename: str, func_name: str):
        path = os.path.join(PROJECT_ROOT, "kernels", filename)
        with open(path, "r") as f:
            source = f.read()
        lib, err = self.device.newLibraryWithSource_options_error_(source, None, None)
        if lib is None:
            msg = err.localizedDescription() if err else "Unknown compile failure"
            raise RuntimeError(f"Metal shader compile failed ({filename}): {msg}")
        func = lib.newFunctionWithName_(func_name)
        if func is None:
            raise RuntimeError(f"Kernel '{func_name}' not found in {filename}")
        pso, pso_err = self.device.newComputePipelineStateWithFunction_error_(func, None)
        if pso is None:
            msg = pso_err.localizedDescription() if pso_err else "Unknown pipeline failure"
            raise RuntimeError(f"Pipeline state failed ({filename}/{func_name}): {msg}")
        return pso

    def _create_buffer(self, arr: np.ndarray):
        return self.device.newBufferWithBytes_length_options_(
            arr.tobytes(), len(arr.tobytes()), Metal.MTLResourceStorageModeShared
        )

    def _create_buffer_pool(self) -> None:
        self.buf_residual = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_x = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_attn_normed = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_q = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_k = self.device.newBufferWithLength_options_(K_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_v = self.device.newBufferWithLength_options_(K_DIM * 2, Metal.MTLResourceStorageModeShared)
        self.buf_scores = self.device.newBufferWithLength_options_(NUM_Q_HEADS * 4, Metal.MTLResourceStorageModeShared)
        self.buf_attn_raw = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_attn_out = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_mlp_normed = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_gate_out = self.device.newBufferWithLength_options_(self.mlp_intermediate * 2, Metal.MTLResourceStorageModeShared)
        self.buf_up_out = self.device.newBufferWithLength_options_(self.mlp_intermediate * 2, Metal.MTLResourceStorageModeShared)
        self.buf_swiglu_out = self.device.newBufferWithLength_options_(self.mlp_intermediate * 2, Metal.MTLResourceStorageModeShared)
        self.buf_mlp_out = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)

        gate_sample = self.layer_weights[0]["gate"]
        down_sample = self.layer_weights[0]["down"]
        k_sample = self.layer_weights[0]["k"]
        self.buf_packed_gate = self.device.newBufferWithLength_options_(len(gate_sample), Metal.MTLResourceStorageModeShared)
        self.buf_packed_up = self.device.newBufferWithLength_options_(len(gate_sample), Metal.MTLResourceStorageModeShared)
        self.buf_packed_down = self.device.newBufferWithLength_options_(len(down_sample), Metal.MTLResourceStorageModeShared)
        self.buf_packed_q = self.device.newBufferWithLength_options_(len(self.layer_weights[0]["q"]), Metal.MTLResourceStorageModeShared)
        self.buf_packed_k = self.device.newBufferWithLength_options_(len(k_sample), Metal.MTLResourceStorageModeShared)
        self.buf_packed_v = self.device.newBufferWithLength_options_(len(k_sample), Metal.MTLResourceStorageModeShared)
        self.buf_packed_o = self.device.newBufferWithLength_options_(len(self.layer_weights[0]["o"]), Metal.MTLResourceStorageModeShared)
        self.buf_scales_gate = self.device.newBufferWithLength_options_(self.mlp_intermediate * 48 * 2, Metal.MTLResourceStorageModeShared)
        self.buf_scales_up = self.device.newBufferWithLength_options_(self.mlp_intermediate * 48 * 2, Metal.MTLResourceStorageModeShared)
        self.buf_scales_down = self.device.newBufferWithLength_options_(self.hidden_dim * 280 * 2, Metal.MTLResourceStorageModeShared)
        self.buf_scales_q = self.device.newBufferWithLength_options_(self.hidden_dim * 48 * 2, Metal.MTLResourceStorageModeShared)
        self.buf_scales_k = self.device.newBufferWithLength_options_(K_DIM * 48 * 2, Metal.MTLResourceStorageModeShared)
        self.buf_scales_v = self.device.newBufferWithLength_options_(K_DIM * 48 * 2, Metal.MTLResourceStorageModeShared)
        self.buf_scales_o = self.device.newBufferWithLength_options_(self.hidden_dim * 48 * 2, Metal.MTLResourceStorageModeShared)
        self.buf_input_norm = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_post_norm = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_logits = self.device.newBufferWithLength_options_(VOCAB_SIZE * 2, Metal.MTLResourceStorageModeShared)

    def _upload_layer_weights(self, layer_idx: int) -> None:
        w = self.layer_weights[layer_idx]
        for name, buf, scale_buf in [
            ("gate", self.buf_packed_gate, self.buf_scales_gate),
            ("up", self.buf_packed_up, self.buf_scales_up),
            ("down", self.buf_packed_down, self.buf_scales_down),
            ("q", self.buf_packed_q, self.buf_scales_q),
            ("k", self.buf_packed_k, self.buf_scales_k),
            ("v", self.buf_packed_v, self.buf_scales_v),
            ("o", self.buf_packed_o, self.buf_scales_o),
        ]:
            dest = np.frombuffer(buf.contents().as_buffer(len(w[name])), dtype=np.uint8)
            dest[:] = w[name]
            scale_key = f"{name}_scales"
            dest = np.frombuffer(scale_buf.contents().as_buffer(len(w[scale_key]) * 2), dtype=np.uint8)
            dest[:] = w[scale_key].view(np.uint8)
        dest = np.frombuffer(self.buf_input_norm.contents().as_buffer(len(w["input_norm"]) * 2), dtype=np.uint8)
        dest[:] = w["input_norm"].view(np.uint8)
        dest = np.frombuffer(self.buf_post_norm.contents().as_buffer(len(w["post_norm"]) * 2), dtype=np.uint8)
        dest[:] = w["post_norm"].view(np.uint8)

    def run_inference(self, input_vector: np.ndarray) -> np.ndarray:
        arr = input_vector.astype(np.float16)
        if arr.size != self.hidden_dim:
            raise ValueError(f"Expected input size {self.hidden_dim}, got {arr.size}")
        dest = np.frombuffer(self.buf_residual.contents().as_buffer(self.hidden_dim * 2), dtype=np.uint8)
        dest[:] = arr.view(np.uint8)

        for i in range(self.layers):
            self._execute_transformer_block(i)

        # Global RMSNorm (model.norm.weight) before classifier
        cmd_buf = self.cmd_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self.kernels["norm"])
        encoder.setBuffer_offset_atIndex_(self.buf_residual, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_final_norm, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_x, 0, 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))

        # LM Head: 1536 -> 151936 (vocab logits). K=1536, Rows=151936.
        encoder.setComputePipelineState_(self.kernels["gemv"])
        encoder.setBuffer_offset_atIndex_(self.buf_lm_head_packed, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_lm_head_scales, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_x, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_logits, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(VOCAB_SIZE, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        result = np.frombuffer(self.buf_logits.contents().as_buffer(VOCAB_SIZE * 2), dtype=np.float16).copy()
        return result

    def _execute_transformer_block(self, layer_idx: int) -> None:
        self._upload_layer_weights(layer_idx)

        cmd_buf = self.cmd_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        # buf_residual = block input. No CPU copy: residual add writes block output to buf_residual for next layer.

        # 1. Input norm: buf_residual -> buf_attn_normed
        encoder.setComputePipelineState_(self.kernels["norm"])
        encoder.setBuffer_offset_atIndex_(self.buf_residual, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_input_norm, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_attn_normed, 0, 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))

        # 2. Q,K,V projections
        encoder.setComputePipelineState_(self.kernels["gemv"])
        encoder.setBuffer_offset_atIndex_(self.buf_packed_q, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_scales_q, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_attn_normed, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_q, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        encoder.setBuffer_offset_atIndex_(self.buf_packed_k, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_scales_k, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_attn_normed, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_k, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(K_DIM, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        encoder.setBuffer_offset_atIndex_(self.buf_packed_v, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_scales_v, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_attn_normed, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_v, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(K_DIM, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        # 3. RoPE on Q and K (in-place)
        encoder.setComputePipelineState_(self.kernels["rope"])
        encoder.setBuffer_offset_atIndex_(self.buf_q, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim // 2, 1, 1), Metal.MTLSize(256, 1, 1))
        encoder.setBuffer_offset_atIndex_(self.buf_k, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(K_DIM // 2, 1, 1), Metal.MTLSize(128, 1, 1))

        # 4. Attention scores: Q @ K^T
        encoder.setComputePipelineState_(self.kernels["attn"])
        encoder.setBuffer_offset_atIndex_(self.buf_q, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_k, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_scores, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_seq_len, 0, 3)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(NUM_Q_HEADS, 1, 1), Metal.MTLSize(NUM_Q_HEADS, 1, 1))

        # 5. Softmax (12 heads)
        encoder.setComputePipelineState_(self.kernels["softmax"])
        encoder.setBuffer_offset_atIndex_(self.buf_scores, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_seq_len, 0, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(NUM_Q_HEADS, 1, 1), Metal.MTLSize(NUM_Q_HEADS, 1, 1))

        # 6. Attn weighted sum: scores @ V
        encoder.setComputePipelineState_(self.kernels["attn_sum"])
        encoder.setBuffer_offset_atIndex_(self.buf_scores, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_v, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_attn_raw, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_seq_len, 0, 3)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(6, 1, 1), Metal.MTLSize(256, 1, 1))

        # 7. O projection
        encoder.setComputePipelineState_(self.kernels["gemv"])
        encoder.setBuffer_offset_atIndex_(self.buf_packed_o, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_scales_o, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_attn_raw, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_attn_out, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        # 8. Attention residual: buf_x = buf_residual + buf_attn_out
        encoder.setComputePipelineState_(self.kernels["resid"])
        encoder.setBuffer_offset_atIndex_(self.buf_residual, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_attn_out, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_x, 0, 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))

        # 9. Post-attention norm: buf_x -> buf_mlp_normed
        encoder.setComputePipelineState_(self.kernels["norm"])
        encoder.setBuffer_offset_atIndex_(self.buf_x, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_post_norm, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_mlp_normed, 0, 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))

        # 10. MLP: Gate, Up, SwiGLU, Down
        encoder.setComputePipelineState_(self.kernels["gemv"])
        encoder.setBuffer_offset_atIndex_(self.buf_packed_gate, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_scales_gate, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_mlp_normed, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_gate_out, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(self.mlp_intermediate, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        encoder.setBuffer_offset_atIndex_(self.buf_packed_up, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_scales_up, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_mlp_normed, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_up_out, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(self.mlp_intermediate, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        encoder.setComputePipelineState_(self.kernels["swiglu"])
        encoder.setBuffer_offset_atIndex_(self.buf_gate_out, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_up_out, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_swiglu_out, 0, 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.mlp_intermediate, 1, 1), Metal.MTLSize(768, 1, 1))

        encoder.setComputePipelineState_(self.kernels["gemv"])
        encoder.setBuffer_offset_atIndex_(self.buf_packed_down, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_scales_down, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_swiglu_out, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buf_mlp_out, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.buf_const_8960, 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        # 11. MLP residual: buf_residual = buf_x + buf_mlp_out (block output for next layer)
        encoder.setComputePipelineState_(self.kernels["resid"])
        encoder.setBuffer_offset_atIndex_(self.buf_x, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buf_mlp_out, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buf_residual, 0, 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()


if __name__ == "__main__":
    weights_dir = os.path.join(PROJECT_ROOT, "weights")
    engine = QwenEngine(weights_dir)
    vec = np.random.randn(HIDDEN_DIM).astype(np.float16)
    result = engine.run_inference(vec)
    print(f"Result (first 5): {result[:5]}")
