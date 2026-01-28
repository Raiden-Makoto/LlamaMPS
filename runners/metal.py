import Metal # type: ignore
import numpy as np # type: ignore
import os # type: ignore

# --- 1. Setup Metal Hardware ---
device = Metal.MTLCreateSystemDefaultDevice()
command_queue = device.newCommandQueue()

# --- 2. Load the "Fuel" ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

# Weight Paths (assume pack_weights.py has been run for q_proj, k_proj, v_proj)
PACKED_Q_PATH = os.path.join(WEIGHTS_DIR, "q_proj_4bit.bin")
PACKED_K_PATH = os.path.join(WEIGHTS_DIR, "k_proj_4bit.bin")
PACKED_V_PATH = os.path.join(WEIGHTS_DIR, "v_proj_4bit.bin")
SCALES_Q_PATH = os.path.join(WEIGHTS_DIR, "q_proj_scales.bin")
SCALES_K_PATH = os.path.join(WEIGHTS_DIR, "k_proj_scales.bin")
SCALES_V_PATH = os.path.join(WEIGHTS_DIR, "v_proj_scales.bin")
NORM_PATH = os.path.join(WEIGHTS_DIR, "layer0_norm.bin")

# Load Data
packed_q = np.fromfile(PACKED_Q_PATH, dtype=np.uint8)
packed_k = np.fromfile(PACKED_K_PATH, dtype=np.uint8)
packed_v = np.fromfile(PACKED_V_PATH, dtype=np.uint8)
scales_q = np.fromfile(SCALES_Q_PATH, dtype=np.float16)
scales_k = np.fromfile(SCALES_K_PATH, dtype=np.float16)
scales_v = np.fromfile(SCALES_V_PATH, dtype=np.float16)
norm_weights = np.fromfile(NORM_PATH, dtype=np.float16)
input_vec = np.random.randn(1536).astype(np.float16)

# --- 3. Create GPU Buffers ---
def create_buffer(arr):
    return device.newBufferWithBytes_length_options_(
        arr.tobytes(), len(arr.tobytes()), Metal.MTLResourceStorageModeShared
    )

buf_packed_q = create_buffer(packed_q)
buf_packed_k = create_buffer(packed_k)
buf_packed_v = create_buffer(packed_v)
buf_scales_q = create_buffer(scales_q)
buf_scales_k = create_buffer(scales_k)
buf_scales_v = create_buffer(scales_v)
buf_norm = create_buffer(norm_weights)
buf_input = create_buffer(input_vec)

# Intermediate: Input -> RMSNorm -> buf_normed (shared input for Q,K,V)
buf_normed = device.newBufferWithLength_options_(1536 * 2, Metal.MTLResourceStorageModeShared)
# Outputs: Q 1536, K 256, V 256
buf_q_out = device.newBufferWithLength_options_(1536 * 2, Metal.MTLResourceStorageModeShared)
buf_k_out = device.newBufferWithLength_options_(256 * 2, Metal.MTLResourceStorageModeShared)
buf_v_out = device.newBufferWithLength_options_(256 * 2, Metal.MTLResourceStorageModeShared)

# --- 4. Compile Shaders ---
def get_pso(filename, func_name):
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

# --- 5. The Chain Execution ---
cmd_buf = command_queue.commandBuffer()
encoder = cmd_buf.computeCommandEncoder()

# Pass 1: RMSNorm
encoder.setComputePipelineState_(pso_norm)
encoder.setBuffer_offset_atIndex_(buf_input, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_norm,  0, 1)
encoder.setBuffer_offset_atIndex_(buf_normed, 0, 2)
encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(1536, 1, 1), Metal.MTLSize(768, 1, 1))

# Pass 2: Triple Projection (Q, K, V)
encoder.setComputePipelineState_(pso_gemv)
# Shared input for all three: buf_normed (buffer 2). Buffers 0,1,3 are per-projection.

# 1. Project Q (1536 rows)
encoder.setBuffer_offset_atIndex_(buf_packed_q, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_scales_q, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_normed, 0, 2)
encoder.setBuffer_offset_atIndex_(buf_q_out, 0, 3)
encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(1536, 1, 1), Metal.MTLSize(768, 1, 1))

# 2. Project K (256 rows)
encoder.setBuffer_offset_atIndex_(buf_packed_k, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_scales_k, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_normed, 0, 2)
encoder.setBuffer_offset_atIndex_(buf_k_out, 0, 3)
encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(256, 1, 1), Metal.MTLSize(768, 1, 1))

# 3. Project V (256 rows)
encoder.setBuffer_offset_atIndex_(buf_packed_v, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_scales_v, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_normed, 0, 2)
encoder.setBuffer_offset_atIndex_(buf_v_out, 0, 3)
encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(256, 1, 1), Metal.MTLSize(768, 1, 1))

encoder.endEncoding()
cmd_buf.commit()
cmd_buf.waitUntilCompleted()

# --- 6. Results ---
q_out = np.frombuffer(buf_q_out.contents().as_buffer(1536 * 2), dtype=np.float16)
k_out = np.frombuffer(buf_k_out.contents().as_buffer(256 * 2), dtype=np.float16)
v_out = np.frombuffer(buf_v_out.contents().as_buffer(256 * 2), dtype=np.float16)
print("Pipeline Complete.")
print(f"Q (first 5): {q_out[:5]}")
print(f"K (first 5): {k_out[:5]}")
print(f"V (first 5): {v_out[:5]}")