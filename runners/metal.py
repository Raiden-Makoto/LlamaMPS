import Metal # type: ignore
import numpy as np # type: ignore
import os # type: ignore

# --- 1. Setup Metal Hardware ---
device = Metal.MTLCreateSystemDefaultDevice()
if not device:
    raise RuntimeError("Metal is not supported on this system.")
command_queue = device.newCommandQueue()

# --- 2. Load the "Fuel" ---
# We'll test a single "slice" of the Qwen q_proj layer (1536 elements)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
PACKED_PATH = os.path.join(WEIGHTS_DIR, "q_proj_4bit.bin")
SCALES_PATH = os.path.join(WEIGHTS_DIR, "q_proj_scales.bin")

if not os.path.exists(PACKED_PATH):
    raise FileNotFoundError("Run pack_weights.py first to generate .bin files!")

# Load raw bytes: 1536 weights = 768 bytes (packed)
packed_weights = np.fromfile(PACKED_PATH, dtype=np.uint8)[:768]
# 1536 weights / 32 per block = 48 scales
scales = np.fromfile(SCALES_PATH, dtype=np.float16)[:48]
# Generate a random input vector for the test
input_vec = np.random.randn(1536).astype(np.float16)

# --- 3. Create GPU Buffers (Unified Memory) ---
def create_buffer(arr):
    # Using MTLResourceStorageModeShared allows CPU and GPU to share this memory
    return device.newBufferWithBytes_length_options_(
        arr.tobytes(), len(arr.tobytes()), Metal.MTLResourceStorageModeShared
    )

buf_weights = create_buffer(packed_weights)
buf_scales  = create_buffer(scales)
buf_input   = create_buffer(input_vec)
# Room for 768 partial sums (one per thread)
buf_output  = device.newBufferWithLength_options_(768 * 2, Metal.MTLResourceStorageModeShared)

# --- 4. Compile the Shader ---
with open("kernels/quant_matmul.metal", "r") as f:
    source = f.read()

library, _ = device.newLibraryWithSource_options_error_(source, None, None)
kernel = library.newFunctionWithName_("dequant_dot_product")
pso, _ = device.newComputePipelineStateWithFunction_error_(kernel, None)

# --- 5. Dispatch to GPU ---
cmd_buf = command_queue.commandBuffer()
encoder = cmd_buf.computeCommandEncoder()
encoder.setComputePipelineState_(pso)

# Indexing matches our kernel [[buffer(X)]]
encoder.setBuffer_offset_atIndex_(buf_weights, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_scales, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_input, 0, 2)
encoder.setBuffer_offset_atIndex_(buf_output, 0, 3)

# We run 1 thread per byte
grid_size = Metal.MTLSize(768, 1, 1)
thread_group_size = Metal.MTLSize(min(768, pso.maxTotalThreadsPerThreadgroup()), 1, 1)
encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)

encoder.endEncoding()
cmd_buf.commit()
cmd_buf.waitUntilCompleted()

# --- 6. The Ground Truth Comparison (Fixed) ---
gpu_out_ptr = buf_output.contents().as_buffer(768 * 2)
gpu_res = np.frombuffer(gpu_out_ptr, dtype=np.float16).sum()

print(f"--- Result Verification ---")
print(f"GPU 4-bit Result: {gpu_res:.4f}")

# Manual Python Check with Signed Casting
cpu_res = 0.0
for i in range(768):
    byte = int(packed_weights[i]) # Cast to standard Python int to avoid numpy uint8 wrapping
    
    # Extract bits
    low_bits = byte & 0x0F
    high_bits = byte >> 4
    
    # Convert to signed range [-8, 7]
    w1 = float(low_bits) - 8.0
    w2 = float(high_bits) - 8.0
    
    scale = float(scales[i // 16])
    
    # Accumulate
    cpu_res += (w1 * scale * float(input_vec[i*2])) + (w2 * scale * float(input_vec[i*2+1]))

print(f"CPU Reference: {cpu_res:.4f}")
# Accuracy check (using a slightly higher tolerance for half-precision drift)
match = np.isclose(gpu_res, cpu_res, rtol=1e-2)
print(f"Accuracy Match: {match}")