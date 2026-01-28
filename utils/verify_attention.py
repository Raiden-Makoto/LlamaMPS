import numpy as np # type: ignore

def verify_attention_indexing(q_vec, k_vec):
    # Head dim is 128
    # Q has 12 heads, K has 2 heads
    num_q_heads = 12
    num_k_heads = 2
    heads_per_group = num_q_heads // num_k_heads # 6
    
    cpu_scores = np.zeros(num_q_heads, dtype=np.float32)
    
    for q_idx in range(num_q_heads):
        # 1. Map Q head to the correct K head (The GQA logic)
        k_idx = q_idx // heads_per_group
        
        # 2. Extract the 128-dim vectors
        q_head = q_vec[q_idx * 128 : (q_idx + 1) * 128].astype(np.float32)
        k_head = k_vec[k_idx * 128 : (k_idx + 1) * 128].astype(np.float32)
        
        # 3. Dot Product + Scale (1 / sqrt(128) ≈ 0.088388)
        dot = np.dot(q_head, k_head)
        cpu_scores[q_idx] = dot * 0.088388
        
    return cpu_scores