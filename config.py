# import packages
import torch


GPT_CONFIG = {
    
    "vocab_size": 201088,
    "context_length": 4096,
    "emb_dim": 2880,
    "hid_dim": 2880,
    "head_dim": 64,
    
    "n_heads": 64,
    "n_kv_heads": 8,
    
    "n_layers": 24,
    
    "num_experts": 32,
    "num_active_experts": 4,
    
    "rope_base": 150000.0,
    "sliding_window": 128, 
    "layer_types": [
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        
    ],
    "dtype": torch.bfloat16
}