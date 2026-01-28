import Metal # type: ignore
import numpy as np # type: ignore
import os # type: ignore

# --- 1. Setup Metal Hardware ---
device = Metal.MTLCreateSystemDefaultDevice()
command_queue = device.newCommandQueue()

# --- 2. Load the "Fuel" ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

# Weight Paths
PACKED_PATH = os.path.join(WEIGHTS_DIR, "q_proj_4bit.bin")
SCALES_PATH = os.path.join(WEIGHTS_DIR, "q_proj_scales.bin")
NORM_PATH   = os.path.join(WEIGHTS_DIR, "layer0_norm.bin") # The new gatekeeper weights

# Load Data
packed_weights = np.fromfile(PACKED_PATH, dtype=np.uint8)
scales         = np.fromfile(SCALES_PATH, dtype=np.float16)
norm_weights   = np.fromfile(NORM_PATH, dtype=np.float16)
input_vec      = np.random.randn(1536).astype(np.float16)

# --- 3. Create GPU Buffers ---
def create_buffer(arr):
    return device.newBufferWithBytes_length_options_(
        arr.tobytes(), len(arr.tobytes()), Metal.MTLResourceStorageModeShared
    )

buf_packed = create_buffer(packed_weights)
buf_scales = create_buffer(scales)
buf_norm   = create_buffer(norm_weights)
buf_input  = create_buffer(input_vec)

# Intermediate buffer: Input -> RMSNorm -> [buf_normed]
buf_normed = device.newBufferWithLength_options_(1536 * 2, Metal.MTLResourceStorageModeShared)
# Final result: [buf_normed] -> GEMV -> [buf_output]
buf_output = device.newBufferWithLength_options_(1536 * 2, Metal.MTLResourceStorageModeShared)

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

# Pass 2: GEMV (Matrix Multiply)
encoder.setComputePipelineState_(pso_gemv)
encoder.setBuffer_offset_atIndex_(buf_packed, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_scales, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_normed, 0, 2) # Take output from Pass 1
encoder.setBuffer_offset_atIndex_(buf_output, 0, 3)
encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(1536, 1, 1), Metal.MTLSize(768, 1, 1))

encoder.endEncoding()
cmd_buf.commit()
cmd_buf.waitUntilCompleted()

# --- 6. Results ---
final_output = np.frombuffer(buf_output.contents().as_buffer(1536 * 2), dtype=np.float16)
print(f"Pipeline Complete.")
print(f"Normalized Projection (First 5): {final_output[:5]}")