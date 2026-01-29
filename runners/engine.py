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

# GEMV dispatch: aligned with the 768 threads / 24 SIMD groups logic in quant_matmul.metal
GEMV_THREADS = 768
GEMV_SIMD_GROUPS = GEMV_THREADS // 32

def load_bin(path: str, dtype=np.uint8) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing weight file: {path}")
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

        # 1. Preload Weights into CPU RAM
        self.layer_weights = self._preload_weights()

        # 2. Global Weights (Start & End)
        global_dir = os.path.join(self.weights_dir, "global")
        final_norm = load_bin(os.path.join(global_dir, "final_norm.bin"), np.float16)
        self.buf_final_norm = self._create_buffer(final_norm)

        # LM Head (4-bit tied weights)
        self.buf_lm_head_packed = self._create_buffer(load_bin(os.path.join(self.weights_dir, "embed_tokens_4bit.bin")))
        self.buf_lm_head_scales = self._create_buffer(load_bin(os.path.join(self.weights_dir, "embed_tokens_scales.bin"), np.float16))

        # 3. Compile Shaders
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

        # 4. Initialize Buffer Pool and Constants
        self._create_buffer_pool()
        self.buf_const_1536 = self._create_buffer(np.array([1536, GEMV_SIMD_GROUPS, GEMV_THREADS], dtype=np.uint32))
        self.buf_const_8960 = self._create_buffer(np.array([8960, GEMV_SIMD_GROUPS, GEMV_THREADS], dtype=np.uint32))
        self.buf_seq_len = self._create_buffer(np.array([1], dtype=np.uint32))

    def _get_pso(self, filename: str, func_name: str):
        path = os.path.join(PROJECT_ROOT, "kernels", filename)
        with open(path, "r") as f:
            source = f.read()
        lib, err = self.device.newLibraryWithSource_options_error_(source, None, None)
        if lib is None:
            raise RuntimeError(f"Metal compile failed ({filename}): {err.localizedDescription()}")
        func = lib.newFunctionWithName_(func_name)
        pso, pso_err = self.device.newComputePipelineStateWithFunction_error_(func, None)
        if pso is None:
            raise RuntimeError(f"Pipeline failed ({filename}): {pso_err.localizedDescription()}")
        return pso

    def _create_buffer(self, arr: np.ndarray):
        return self.device.newBufferWithBytes_length_options_(
            arr.tobytes(), len(arr.tobytes()), Metal.MTLResourceStorageModeShared
        )

    def _create_buffer_pool(self) -> None:
        """Reusable intermediate buffers to minimize VRAM fragmentation."""
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
        self.buf_logits = self.device.newBufferWithLength_options_(VOCAB_SIZE * 2, Metal.MTLResourceStorageModeShared)

        # Weight Transfer Buffers
        self.buf_packed_q = self.device.newBufferWithLength_options_(self.hidden_dim * (self.hidden_dim // 2), Metal.MTLResourceStorageModeShared)
        self.buf_scales_q = self.device.newBufferWithLength_options_(self.hidden_dim * (self.hidden_dim // 32) * 2, Metal.MTLResourceStorageModeShared)
        self.buf_packed_k = self.device.newBufferWithLength_options_(K_DIM * (self.hidden_dim // 2), Metal.MTLResourceStorageModeShared)
        self.buf_scales_k = self.device.newBufferWithLength_options_(K_DIM * (self.hidden_dim // 32) * 2, Metal.MTLResourceStorageModeShared)
        self.buf_packed_v = self.device.newBufferWithLength_options_(K_DIM * (self.hidden_dim // 2), Metal.MTLResourceStorageModeShared)
        self.buf_scales_v = self.device.newBufferWithLength_options_(K_DIM * (self.hidden_dim // 32) * 2, Metal.MTLResourceStorageModeShared)
        self.buf_packed_o = self.device.newBufferWithLength_options_(self.hidden_dim * (self.hidden_dim // 2), Metal.MTLResourceStorageModeShared)
        self.buf_scales_o = self.device.newBufferWithLength_options_(self.hidden_dim * (self.hidden_dim // 32) * 2, Metal.MTLResourceStorageModeShared)
        self.buf_packed_gate = self.device.newBufferWithLength_options_(self.mlp_intermediate * (self.hidden_dim // 2), Metal.MTLResourceStorageModeShared)
        self.buf_scales_gate = self.device.newBufferWithLength_options_(self.mlp_intermediate * (self.hidden_dim // 32) * 2, Metal.MTLResourceStorageModeShared)
        self.buf_packed_up = self.device.newBufferWithLength_options_(self.mlp_intermediate * (self.hidden_dim // 2), Metal.MTLResourceStorageModeShared)
        self.buf_scales_up = self.device.newBufferWithLength_options_(self.mlp_intermediate * (self.hidden_dim // 32) * 2, Metal.MTLResourceStorageModeShared)
        self.buf_packed_down = self.device.newBufferWithLength_options_(self.hidden_dim * (self.mlp_intermediate // 2), Metal.MTLResourceStorageModeShared)
        self.buf_scales_down = self.device.newBufferWithLength_options_(self.hidden_dim * (self.mlp_intermediate // 32) * 2, Metal.MTLResourceStorageModeShared)
        self.buf_input_norm = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)
        self.buf_post_norm = self.device.newBufferWithLength_options_(self.hidden_dim * 2, Metal.MTLResourceStorageModeShared)

    def _preload_weights(self) -> list:
        weights = []
        for i in range(self.layers):
            d = os.path.join(self.weights_dir, f"layer{i}")
            w = {
                "in_norm": load_bin(os.path.join(d, f"layer{i}_norm.bin"), np.float16),
                "post_norm": load_bin(os.path.join(d, f"layer{i}_post_attn_norm.bin"), np.float16),
                "q": load_bin(os.path.join(d, "q_proj_4bit.bin")),
                "qs": load_bin(os.path.join(d, "q_proj_scales.bin"), np.float16),
                "k": load_bin(os.path.join(d, "k_proj_4bit.bin")),
                "ks": load_bin(os.path.join(d, "k_proj_scales.bin"), np.float16),
                "v": load_bin(os.path.join(d, "v_proj_4bit.bin")),
                "vs": load_bin(os.path.join(d, "v_proj_scales.bin"), np.float16),
                "o": load_bin(os.path.join(d, "o_proj_4bit.bin")),
                "os": load_bin(os.path.join(d, "o_proj_scales.bin"), np.float16),
                "gt": load_bin(os.path.join(d, "gate_proj_4bit.bin")),
                "gts": load_bin(os.path.join(d, "gate_proj_scales.bin"), np.float16),
                "up": load_bin(os.path.join(d, "up_proj_4bit.bin")),
                "ups": load_bin(os.path.join(d, "up_proj_scales.bin"), np.float16),
                "dn": load_bin(os.path.join(d, "down_proj_4bit.bin")),
                "dns": load_bin(os.path.join(d, "down_proj_scales.bin"), np.float16),
            }
            weights.append(w)
        return weights

    def _upload_layer(self, idx: int):
        w = self.layer_weights[idx]
        def up(buf, data): np.copyto(np.frombuffer(buf.contents().as_buffer(len(data.tobytes())), dtype=np.uint8), data.view(np.uint8))
        up(self.buf_input_norm, w["in_norm"]); up(self.buf_post_norm, w["post_norm"])
        up(self.buf_packed_q, w["q"]); up(self.buf_scales_q, w["qs"])
        up(self.buf_packed_k, w["k"]); up(self.buf_scales_k, w["ks"])
        up(self.buf_packed_v, w["v"]); up(self.buf_scales_v, w["vs"])
        up(self.buf_packed_o, w["o"]); up(self.buf_scales_o, w["os"])
        up(self.buf_packed_gate, w["gt"]); up(self.buf_scales_gate, w["gts"])
        up(self.buf_packed_up, w["up"]); up(self.buf_scales_up, w["ups"])
        up(self.buf_packed_down, w["dn"]); up(self.buf_scales_down, w["dns"])

    def run_inference(self, input_vector: np.ndarray) -> np.ndarray:
        np.copyto(np.frombuffer(self.buf_residual.contents().as_buffer(self.hidden_dim * 2), dtype=np.float16), input_vector.astype(np.float16))
        
        for i in range(self.layers):
            self._execute_block(i)

        # Final Stage: Global Norm -> LM Head
        cmd = self.cmd_queue.commandBuffer(); enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.kernels["norm"])
        enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 0)
        enc.setBuffer_offset_atIndex_(self.buf_final_norm, 0, 1)
        enc.setBuffer_offset_atIndex_(self.buf_x, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))

        enc.setComputePipelineState_(self.kernels["gemv"])
        enc.setBuffer_offset_atIndex_(self.buf_lm_head_packed, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_lm_head_scales, 0, 1)
        enc.setBuffer_offset_atIndex_(self.buf_x, 0, 2); enc.setBuffer_offset_atIndex_(self.buf_logits, 0, 3)
        enc.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(VOCAB_SIZE, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        
        return np.frombuffer(self.buf_logits.contents().as_buffer(VOCAB_SIZE * 2), dtype=np.float16).copy()

    def predict_next_token(self, logits: np.ndarray) -> int:
        return int(np.argmax(logits.astype(np.float32)))

    def _execute_block(self, idx: int):
        self._upload_layer(idx)
        cmd = self.cmd_queue.commandBuffer(); enc = cmd.computeCommandEncoder()
        
        # Norm & QKV
        enc.setComputePipelineState_(self.kernels["norm"])
        enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_input_norm, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_attn_normed, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))

        enc.setComputePipelineState_(self.kernels["gemv"])
        for p, s, b, c in [(self.buf_packed_q, self.buf_scales_q, self.buf_q, self.hidden_dim), (self.buf_packed_k, self.buf_scales_k, self.buf_k, K_DIM), (self.buf_packed_v, self.buf_scales_v, self.buf_v, K_DIM)]:
            enc.setBuffer_offset_atIndex_(p, 0, 0); enc.setBuffer_offset_atIndex_(s, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_attn_normed, 0, 2); enc.setBuffer_offset_atIndex_(b, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(c, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        # RoPE, Attn, Softmax, Sum
        enc.setComputePipelineState_(self.kernels["rope"])
        enc.setBuffer_offset_atIndex_(self.buf_q, 0, 0); enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(768, 1, 1), Metal.MTLSize(256, 1, 1))
        enc.setBuffer_offset_atIndex_(self.buf_k, 0, 0); enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(128, 1, 1), Metal.MTLSize(128, 1, 1))

        enc.setComputePipelineState_(self.kernels["attn"])
        enc.setBuffer_offset_atIndex_(self.buf_q, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_k, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_scores, 0, 2); enc.setBuffer_offset_atIndex_(self.buf_seq_len, 0, 3)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(12, 1, 1), Metal.MTLSize(12, 1, 1))

        enc.setComputePipelineState_(self.kernels["softmax"])
        enc.setBuffer_offset_atIndex_(self.buf_scores, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_seq_len, 0, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(12, 1, 1), Metal.MTLSize(12, 1, 1))

        enc.setComputePipelineState_(self.kernels["attn_sum"])
        enc.setBuffer_offset_atIndex_(self.buf_scores, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_v, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_attn_raw, 0, 2); enc.setBuffer_offset_atIndex_(self.buf_seq_len, 0, 3)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(6, 1, 1), Metal.MTLSize(256, 1, 1))

        # O-Proj & Attn Resid
        enc.setComputePipelineState_(self.kernels["gemv"])
        enc.setBuffer_offset_atIndex_(self.buf_packed_o, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_scales_o, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_attn_raw, 0, 2); enc.setBuffer_offset_atIndex_(self.buf_attn_out, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        enc.setComputePipelineState_(self.kernels["resid"])
        enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_attn_out, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_x, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))

        # Post-Norm & MLP
        enc.setComputePipelineState_(self.kernels["norm"])
        enc.setBuffer_offset_atIndex_(self.buf_x, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_post_norm, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_mlp_normed, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))

        enc.setComputePipelineState_(self.kernels["gemv"])
        for p, s, b in [(self.buf_packed_gate, self.buf_scales_gate, self.buf_gate_out), (self.buf_packed_up, self.buf_scales_up, self.buf_up_out)]:
            enc.setBuffer_offset_atIndex_(p, 0, 0); enc.setBuffer_offset_atIndex_(s, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_mlp_normed, 0, 2); enc.setBuffer_offset_atIndex_(b, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_const_1536, 0, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(self.mlp_intermediate, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        enc.setComputePipelineState_(self.kernels["swiglu"])
        enc.setBuffer_offset_atIndex_(self.buf_gate_out, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_up_out, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_swiglu_out, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.mlp_intermediate, 1, 1), Metal.MTLSize(768, 1, 1))

        enc.setComputePipelineState_(self.kernels["gemv"])
        enc.setBuffer_offset_atIndex_(self.buf_packed_down, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_scales_down, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_swiglu_out, 0, 2); enc.setBuffer_offset_atIndex_(self.buf_mlp_out, 0, 3); enc.setBuffer_offset_atIndex_(self.buf_const_8960, 0, 4)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(GEMV_THREADS, 1, 1))

        enc.setComputePipelineState_(self.kernels["resid"])
        enc.setBuffer_offset_atIndex_(self.buf_x, 0, 0); enc.setBuffer_offset_atIndex_(self.buf_mlp_out, 0, 1); enc.setBuffer_offset_atIndex_(self.buf_residual, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.hidden_dim, 1, 1), Metal.MTLSize(768, 1, 1))
        enc.endEncoding(); cmd.commit()
        cmd.waitUntilCompleted()