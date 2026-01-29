import numpy as np #type: ignore

def verify_swiglu(gate_out, up_out):
    """
    CPU Reference for SwiGLU: (SiLU(gate) * up)
    gate_out: output from W1 (8960 dim)
    up_out: output from W3 (8960 dim)
    """
    # Convert to float32 for high-precision verification
    x = gate_out.astype(np.float32)
    y = up_out.astype(np.float32)
    
    # 1. Calculate Sigmoid
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    
    # 2. Calculate SiLU(x) * y
    # silu = x * sigmoid
    cpu_swiglu = (x * sigmoid) * y
    
    return cpu_swiglu.astype(np.float16)