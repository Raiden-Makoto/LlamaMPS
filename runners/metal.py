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
kernel_path = os.path.join(PROJECT_ROOT, "kernels", "quant_matmul.metal")
with open(kernel_path, "r") as f:
    source = f.read()

library, _ = device.newLibraryWithSource_options_error_(source, None, None)
kernel = library.newFunctionWithName_("gemv_q4_0")
pso, _ = device.newComputePipelineStateWithFunction_error_(kernel, None)

# --- 5. Updated Dispatch for Full Matrix ---
# We want to process 1536 rows
num_rows = 1536
# Each row needs 768 threads to process its 1536 weights (2 per thread)
threads_per_row = 768

# Create an output buffer for the whole vector
buf_output = device.newBufferWithLength_options_(
    num_rows * 2, # 1536 elements * 2 bytes (float16)
    Metal.MTLResourceStorageModeShared
)

cmd_buf = command_queue.commandBuffer()
encoder = cmd_buf.computeCommandEncoder()
encoder.setComputePipelineState_(pso)

encoder.setBuffer_offset_atIndex_(buf_weights, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_scales, 0, 1)
encoder.setBuffer_offset_atIndex_(buf_input, 0, 2)
encoder.setBuffer_offset_atIndex_(buf_output, 0, 3)

# Grid of [Rows] threadgroups, each with [768] threads
grid_size = Metal.MTLSize(num_rows, 1, 1) # This determines 'row'
thread_group_size = Metal.MTLSize(threads_per_row, 1, 1) # This determines 'tid'

# We use dispatchThreadgroups because we defined one group per row
encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, thread_group_size)

encoder.endEncoding()
cmd_buf.commit()
cmd_buf.waitUntilCompleted()

# --- 6. Verification ---
gpu_final_vector = np.frombuffer(buf_output.contents().as_buffer(num_rows * 2), dtype=np.float16)
print(f"GPU Matrix Multiply Complete. Output Shape: {gpu_final_vector.shape}")
print(f"First 5 values: {gpu_final_vector[:5]}")