import Metal  # type: ignore
import numpy as np  # type: ignore
import os  # type: ignore
import sys  # type: ignore


# --- 1. Setup Metal Hardware ---
device = Metal.MTLCreateSystemDefaultDevice()
if device is None:
    raise RuntimeError("Metal is not supported on this system.")
command_queue = device.newCommandQueue()

# --- 2. Load the "Fuel" (Weights) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")


def load_bin(name, dtype=np.uint8):
    return np.fromfile(os.path.join(WEIGHTS_DIR, name), dtype=dtype)


# MLP Projections (Gate=W1, Down=W2, Up=W3)
packed_gate = load_bin("gate_proj_4bit.bin")
packed_up = load_bin("up_proj_4bit.bin")
packed_down = load_bin("down_proj_4bit.bin")
scales_gate = load_bin("gate_proj_scales.bin", np.float16)
scales_up = load_bin("up_proj_scales.bin", np.float16)
scales_down = load_bin("down_proj_scales.bin", np.float16)
norm_1_weights = load_bin("layer0_post_attn_norm.bin", np.float16)  # Post-attention norm

input_vec = np.random.randn(1536).astype(np.float16)


# --- 3. Create GPU Buffers ---
def create_buffer(arr):
    return device.newBufferWithBytes_length_options_(
        arr.tobytes(), len(arr.tobytes()), Metal.MTLResourceStorageModeShared
    )


# MLP-related buffers
buf_packed_gate = create_buffer(packed_gate)
buf_packed_up = create_buffer(packed_up)
buf_packed_down = create_buffer(packed_down)
buf_scales_gate = create_buffer(scales_gate)
buf_scales_up = create_buffer(scales_up)
buf_scales_down = create_buffer(scales_down)
buf_norm_1 = create_buffer(norm_1_weights)

# Treat the input as the pre-MLP residual (attention output + residual)
buf_final_residual = create_buffer(input_vec)

# Intermediate MLP Buffers
buf_mlp_normed = device.newBufferWithLength_options_(1536 * 2, Metal.MTLResourceStorageModeShared)
buf_gate_out = device.newBufferWithLength_options_(8960 * 2, Metal.MTLResourceStorageModeShared)
buf_up_out = device.newBufferWithLength_options_(8960 * 2, Metal.MTLResourceStorageModeShared)
buf_swiglu_out = device.newBufferWithLength_options_(8960 * 2, Metal.MTLResourceStorageModeShared)
buf_mlp_out = device.newBufferWithLength_options_(1536 * 2, Metal.MTLResourceStorageModeShared)
buf_layer_out = device.newBufferWithLength_options_(1536 * 2, Metal.MTLResourceStorageModeShared)


# --- 4. Compile Shaders ---
def get_pso(filename: str, func_name: str):
    """Compile a Metal compute pipeline from a source file and kernel name."""
    path = os.path.join(PROJECT_ROOT, "kernels", filename)
    with open(path, "r") as f:
        source = f.read()
    lib, err = device.newLibraryWithSource_options_error_(source, None, None)
    if lib is None:
        msg = err.localizedDescription() if err else "Unknown compile failure"
        raise RuntimeError(f"Metal shader compile failed ({filename}): {msg}")
    func = lib.newFunctionWithName_(func_name)
    if func is None:
        raise RuntimeError(f"Kernel '{func_name}' not found in {filename}")
    pso, pso_err = device.newComputePipelineStateWithFunction_error_(func, None)
    if pso is None:
        msg = pso_err.localizedDescription() if pso_err else "Unknown pipeline failure"
        raise RuntimeError(f"Pipeline state failed ({filename}/{func_name}): {msg}")
    return pso


pso_norm = get_pso("rms_norm.metal", "rms_norm_q4")
pso_gemv = get_pso("quant_matmul.metal", "gemv_q4_0")
pso_swiglu = get_pso("swiglu.metal", "swiglu")
pso_resid = get_pso("residual.metal", "residual_add")


# --- 5. Phase 2: MLP (The "Knowledge" Block) ---
cmd_buf = command_queue.commandBuffer()
encoder = cmd_buf.computeCommandEncoder()

# 9. Post-Attention Norm
encoder.setComputePipelineState_(pso_norm)
encoder.setBuffer_offset_atIndex_(buf_final_residual, 0, 0) # Input is the Attention-Residual output
encoder.setBuffer_offset_atIndex_(buf_norm_1, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_mlp_normed, 0, 2)
encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(1536, 1, 1), Metal.MTLSize(768, 1, 1))

# 10. MLP Projections (Parallel Gate and Up)
encoder.setComputePipelineState_(pso_gemv)
# Gate (W1)
encoder.setBuffer_offset_atIndex_(buf_packed_gate, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_scales_gate, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_mlp_normed, 0, 2)
encoder.setBuffer_offset_atIndex_(buf_gate_out, 0, 3)
encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(8960, 1, 1), Metal.MTLSize(768, 1, 1))
# Up (W3)
encoder.setBuffer_offset_atIndex_(buf_packed_up, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_scales_up, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_mlp_normed, 0, 2)
encoder.setBuffer_offset_atIndex_(buf_up_out, 0, 3)
encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(8960, 1, 1), Metal.MTLSize(768, 1, 1))

# 11. SwiGLU Activation
encoder.setComputePipelineState_(pso_swiglu)
encoder.setBuffer_offset_atIndex_(buf_gate_out, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_up_out, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_swiglu_out, 0, 2)
encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(8960, 1, 1), Metal.MTLSize(768, 1, 1))

# 12. Down Projection (W2)
encoder.setComputePipelineState_(pso_gemv)
encoder.setBuffer_offset_atIndex_(buf_packed_down, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_scales_down, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_swiglu_out, 0, 2)
encoder.setBuffer_offset_atIndex_(buf_mlp_out, 0, 3)
encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(1536, 1, 1), Metal.MTLSize(768, 1, 1))

# 13. Final Layer Residual Add
encoder.setComputePipelineState_(pso_resid)
encoder.setBuffer_offset_atIndex_(buf_final_residual, 0, 0) # Add back to attention result
encoder.setBuffer_offset_atIndex_(buf_mlp_out, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_layer_out, 0, 2)
encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(1536, 1, 1), Metal.MTLSize(768, 1, 1))

encoder.endEncoding()
cmd_buf.commit()
cmd_buf.waitUntilCompleted()

# --- 7. Final Results ---
layer_0_result = np.frombuffer(buf_layer_out.contents().as_buffer(1536 * 2), dtype=np.float16)
print(f"Layer 0 Complete. Result (first 5): {layer_0_result[:5]}")