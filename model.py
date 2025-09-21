# import packages
import torch
from torch import nn
import math
from torch.nn import functional as F
from config import GPT_CONFIG


# constants
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")



# --- RoPE: rotary positional embeddings ---

# 1) compute sin_thetas & cos_thetas
def compute_rope_params(head_dim: int, base_theta: int, context_length: int, dtype: torch.dtype):
    
    # calc the 2_i vector
    even_indices= torch.arange(0, head_dim, 2, dtype= dtype) # (head_dim / 2)
    
    # calc the inv freq first (omega_i)
    inv_freq= 1 / (base_theta ** (even_indices / head_dim)  ) # (head_dim / 2)
    
    # calc the positions vector
    pos= torch.arange(0, context_length, 1, dtype= dtype) # (s)
    
    # calc the angles or thetas
    thetas= pos.unsqueeze(1) * inv_freq.unsqueeze(0) # (s, 1) * (1, head_dim / 2) = (s, head_dim / 2)
    
    # duplicate to match the head_dim dimension 
    thetas = thetas.repeat_interleave(2, dim=1) # (s, head_dim)

    
    # calc the sin and cos angles
    sin_thetas= torch.sin(thetas) # (s, head_dim)
    cos_thetas= torch.cos(thetas) # (s, head_dim)
    
    return sin_thetas, cos_thetas


# 2) apply rope to the input tensor 
def apply_rope(x: torch.Tensor, sin_thetas: torch.Tensor, cos_thetas: torch.Tensor ):
    
    # x -> (b, n_h, s, d_h)
    # sin_thetas -> (s, d_h)
    # cos_thetas -> (s, d_h)
    
    b, n_h, s, d_h= x.shape
    
    # segerate x into 2 halves
    x1= x[..., :d_h//2] # (b, n_h, s, d_h/2)
    x2= x[..., d_h//2:] # (b, n_h, s, d_h/2)
        
    # expand sin/cos to match x dimensions
    sin_thetas= sin_thetas.unsqueeze(0).unsqueeze(0) # (1, 1, s, d_h)
    cos_thetas= cos_thetas.unsqueeze(0).unsqueeze(0) # (1, 1, s, d_h)
    
    # slice the sin and cos
    sin_thetas= sin_thetas[:, :, :s, :]
    cos_thetas= cos_thetas[:, :, :s, :]
    

    # rotate x
    rot_x= torch.cat([-x2, x1], dim= -1) # (b, n_h, s, d_h)
    
    out= (x * cos_thetas) + (rot_x * sin_thetas)
    
    return out.to(dtype= x.dtype)
    


# --- RMS Normalization ---
class RMSNorm(nn.Module):
    
    def __init__(self, emb_dim: int, eps: float= 1e-6):
        
        super().__init__()
        
        self.eps= eps
        
        self.scale= nn.Parameter(torch.zeros(emb_dim))        
        
        self.shift= nn.Parameter(torch.zeros(emb_dim))        
        
    def forward(self, x: torch.Tensor):
        
        # x -> (B, S, d_m)
        x_f= x.float()
        
        var_x= x_f.pow(2).mean(dim= -1, keepdim= True) 
        rms= torch.sqrt(var_x + self.eps)
        x_norm= x_f / rms
        
        out= x_norm*(1 + self.scale.float()) + self.shift.float();
        
        return out.to(x.dtype) 
        
        
        
        
        
# --- Expert MLP ---
class ExpertMLP(nn.Module):
    
    def __init__(self, emb_dim: int, hid_dim: int, dtype: torch.dtype):
        
        super().__init__()
        
        self.exp_1= nn.Linear(emb_dim, hid_dim, False, dtype= dtype) 
        self.exp_2= nn.Linear(emb_dim, hid_dim, False, dtype= dtype) 
        
        self.contr= nn.Linear(hid_dim, emb_dim, False, dtype= dtype) 
        
        
    def forward(self, x: torch.Tensor):
        
        # x -> (B, S, emb_dim)
        e1= self.exp_1(x) # (B, S, hid_dim)
        e2= self.exp_2(x) # (B, S, hid_dim)
        
        out= F.silu(e1) * e2; # (B, S, hid_dim)
        
        out= self.contr(out) # (B, S, emb_dim)
        
        return out
    
    
    
    
# --- Router MLP ---
class Router(nn.Module):
    def __init__(self, emb_dim: int, num_experts: int, num_active_experts: int, dtype: torch.dtype):
        super().__init__()
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.w_gating = nn.Linear(emb_dim, num_experts, bias=False, dtype= dtype)

    def forward(self, x):
        
        # x -> (b*s, embed_dim)
        print(f"\n\n~~~ Router MLP -> {x.shape}")
        
        logits = self.w_gating(x)
        probs = F.softmax(logits, dim=-1)

        # get top-k experts per token
        scores, indices = torch.topk(probs, self.num_active_experts, dim=-1)
        return scores, indices

    
    
    
    
# --- MoE ---
class MoE(nn.Module):
    def __init__(self, emb_dim: int, hid_dim: int, num_experts: int, num_active_experts: int, dtype: torch.dtype):
        super().__init__()
        
        self.emb_dim= emb_dim
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts

        self.norm = RMSNorm(emb_dim)

        self.experts = nn.ModuleList([
            ExpertMLP(emb_dim, hid_dim, dtype) for _ in range(num_experts)
        ])
        
        self.router = Router(hid_dim, num_experts, num_active_experts, dtype)

    def forward(self, x):
        
        # x -> (b, s, d_in)
        b, s, _= x.shape
        
        x_flat = x.view(-1, self.emb_dim)  # (b*s, d_in)

        # gating + routing
        gating_scores, indices = self.router(x_flat) # (b*s, num_active_experts)
        final_output = torch.zeros_like(x_flat)

        # loop over experts
        for i, expert in enumerate(self.experts):
            
            # mask tokens that go to expert i
            mask = (indices == i).any(dim=-1)   # (b*s,)
            
            if mask.any():
                # tokens for this expert
                expert_in = x_flat[mask]
                expert_out = expert(expert_in)

                # pick correct gating scores for these tokens
                expert_scores = gating_scores[mask, (indices[mask] == i).nonzero(as_tuple=True)[1]]
                expert_scores = expert_scores.unsqueeze(-1)  # (tokens, 1)

                # weighted output
                weighted_out = expert_out * expert_scores

                # scatter back to final_output
                final_output[mask] += weighted_out

        return final_output.view(b, s, self.emb_dim)
    
    
    
    
    
# --- Group Query Attention ---
class GroupQueryAttention(nn.Module):
    
    def __init__(self, d_in: int, n_heads: int, n_kv_heads: int, head_dim: int, dtype: torch.dtype):
        
        super().__init__()
        
        self.d_in= d_in
        self.n_heads= n_heads
        self.n_kv_heads= n_kv_heads
        
        self.head_dim= head_dim
        self.d_out= n_heads * head_dim
        self.kv_groups= self.n_heads // self.n_kv_heads
        
        # projection layers
        self.w_q= nn.Linear(self.d_in, self.d_out, bias= False, dtype= dtype)
        self.w_k= nn.Linear(self.d_in, self.n_kv_heads * self.head_dim, bias= False, dtype= dtype)
        self.w_v= nn.Linear(self.d_in, self.n_kv_heads * self.head_dim, bias= False, dtype= dtype)
        self.w_o= nn.Linear(self.d_out, self.d_in, bias= False, dtype= dtype)
        

        # RMS normalization for q/k
        self.q_norm_layer= RMSNorm(self.head_dim)
        self.k_norm_layer= RMSNorm(self.head_dim)
        
        # sinking
        self.k_sink= nn.Parameter(torch.zeros(self.n_heads, 1, self.head_dim, dtype= dtype)) # (n_heads, 1, head_dim)
        
        nn.init.normal_(self.k_sink, mean=0.0, std=0.02)
        
        
        
        self.scale= self.head_dim ** -0.5 # scaling factor for attention
        
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, sin_thetas: torch.Tensor, cos_thetas: torch.Tensor):
        
        # x -> (b, s, d_in)
        # mask -> (s, s)
        
        b, s, _= x.shape
        
        
        # calc the query, key, value vectors
        q= self.w_q(x) # (b, s, d_out)
        k= self.w_k(x) # (b, s, n_kv_heads * head_dim)
        v= self.w_v(x) # (b, s, n_kv_heads * head_dim)
        
        
        # change shape of query, key, value vectors
        q= q.view(b, s, self.n_heads, self.head_dim) # (b, s, n_heads, head_dim)
        k= k.view(b, s, self.n_kv_heads, self.head_dim) # (b, s, n_kv_heads, head_dim)
        v= v.view(b, s, self.n_kv_heads, self.head_dim) # (b, s, n_kv_heads, head_dim)
        
        
        # change shape of query, key, value vectors
        q= q.transpose(1, 2) # (b, n_heads, s, head_dim)
        k= k.transpose(1, 2) # (b, n_kv_heads, s, head_dim)
        v= v.transpose(1, 2) # (b, n_kv_heads, s, head_dim)
        
        
        # qk norm
        q= self.q_norm_layer(q) # (b, n_heads, s, head_dim)
        k= self.k_norm_layer(k) # (b, n_kv_heads, s, head_dim)
        
        
        # apply RoPE
        q = apply_rope(q, sin_thetas, cos_thetas) # (b, n_heads, s, head_dim)
        k = apply_rope(k, sin_thetas, cos_thetas) # (b, n_kv_heads, s, head_dim)
        
        
        # expand k, v accross all the heads
        k= k.repeat_interleave(self.kv_groups, dim= 1) # (b, n_heads, s, head_dim)
        v= v.repeat_interleave(self.kv_groups, dim= 1) # (b, n_heads, s, head_dim)
        
        # sinking
        k_sink_batch= self.k_sink.unsqueeze(0).expand(b, -1, -1, -1) # (1, n_heads, 1, head_dim) -> (b, n_heads, 1, head_dim)
        
        k= torch.cat([k_sink_batch, k], dim= 2) # (b, n_heads, s+1, head_dim)
        
        
        # update the mask to include the sink
        sink_col = torch.zeros((b, 1, s, 1), device=mask.device, dtype=mask.dtype) # (b, 1, s, 1)

        
        mask= mask.unsqueeze(0).unsqueeze(0).expand(b, -1, -1, -1) # (1, 1, s, s) -> (b, 1, s, s)
        
        mask= torch.cat([sink_col, mask], dim= -1) # (b, 1, s, s + 1)
        
        
        # calc the attention score
        atten_scores= q @ k.transpose(-1, -2) # (b, n_heads, s, head_dim) @ (b, n_heads, head_dim, s + 1) -> (b, n_heads, s, s + 1)
        
        atten_scores*= self.scale # scale the attention score by root of d_m
        
        # apply the mask
        atten_scores.masked_fill_(mask, float("-inf"))
        
        # calc the attention weights
        atten_weights= torch.softmax(atten_scores, dim= -1) # (b, n_heads, s, s + 1)
        
        
        # ignore the sink from the attention weights
        atten_weights= atten_weights[..., 1:] # (b, n_heads, s, s)
        
        # calc the output
        out= atten_weights @ v # (b, n_heads, s, s) @ (b, n_heads, s, head_dim) -> (b, n_heads, s, head_dim)
        
        # change shape of the output: (b, n_heads, s, head_dim) -> (b, s, n_heads, head_dim) -> (B, S, d_out)
        out= out.transpose(1, 2)
        out= out.contiguous().view(b, s, self.d_out)
        
        # pass the output through the final output weight matrix
        out= self.w_o(out) # (B, S, d_in)
        
        return out

    
    
# Transformer Block
class TransformerBlock(nn.Module):
    
    def __init__(self, config: dict):
        
        super().__init__()
        
        self.atten= GroupQueryAttention(
            d_in= config["emb_dim"],
            n_heads= config["n_heads"],
            n_kv_heads= config["n_kv_heads"],
            head_dim= config["head_dim"],
            dtype= config["dtype"]
        )
        
        
        self.moe= MoE(
            emb_dim= config["emb_dim"],
            hid_dim= config["hid_dim"], 
            num_experts= config["num_experts"],
            num_active_experts= config["num_active_experts"],
            dtype= config["dtype"]
        )
        
        self.pre_atten_norm= RMSNorm(emb_dim= config["emb_dim"])
        self.pre_moe_norm= RMSNorm(emb_dim= config["emb_dim"])
        
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, sin_thetas: torch.Tensor, cos_thetas: torch.Tensor):
        
        # attention phase
        shortcut= x
        x= self.pre_atten_norm(x)
        x= self.atten(x, mask, sin_thetas, cos_thetas)
        x= x + shortcut

        # feed-forward phase      
        shortcut= x
        x= self.pre_moe_norm(x)
        x= self.moe(x)
        x= x + shortcut
        
        return x
    
    
    
# Final GPT Model
class GPT(nn.Module):
    
    def __init__(self, config: dict):
        
        super().__init__()
        
        self.vocab_size= config["vocab_size"]   
        self.emb_dim= config["emb_dim"]   
        self.head_dim= config["head_dim"]   
        self.context_length= config["context_length"]   
        self.dtype= config["dtype"]
        self.sliding_window= config["sliding_window"]
        
        
        self.rope_base= config["rope_base"]
        
        
        self.token_embeds= nn.Embedding(self.vocab_size, self.emb_dim, dtype= self.dtype)
        self.blocks= nn.ModuleList([TransformerBlock(config) for _ in range(config["n_layers"])])
        self.final_norm= RMSNorm(self.emb_dim)
        self.out_head= nn.Linear(self.emb_dim, self.vocab_size, bias= False, dtype= self.dtype)
        self.layer_types= config["layer_types"]
        
        
        # compute the sin_thetas & cos_thetas for sliding and full attention
        sin_thetas_sliding, cos_thetas_sliding= compute_rope_params(self.head_dim, self.rope_base, self.context_length, dtype= torch.float32)
        
        sin_thetas_full, cos_thetas_full= compute_rope_params(self.head_dim, self.rope_base, self.context_length, dtype= torch.float32)
        
        
        # register as buffers
        self.register_buffer("sin_thetas_sliding", sin_thetas_sliding, persistent=True)
        self.register_buffer("cos_thetas_sliding", cos_thetas_sliding, persistent=True)
        self.register_buffer("sin_thetas_full", sin_thetas_full, persistent=True)
        self.register_buffer("cos_thetas_full", cos_thetas_full, persistent=True)

           
           
    def _create_masks(self, context_length: int, device):
                
        i= torch.arange(end= context_length, device= device).unsqueeze(1) # (s, 1)
        j= torch.arange(end= context_length, device= device).unsqueeze(0) # (1, s)
        
        # create the mask_full which masks the future tokens
        mask_full= j > i # (s, s)
        mask_past_tokens= (i-j) >= self.sliding_window
        mask_sliding= mask_full | mask_past_tokens # (s, s)
        
        return mask_full, mask_sliding
        
        
    def forward(self, x: torch.Tensor):
        
        # x -> (b, s)
        b, s= x.shape
        
        # get the token embeds
        x= self.token_embeds(x) # (b, s, emb_dim)
        x*= self.emb_dim ** 0.5 # (b, s, emb_dim)
        
        # get the masks
        mask_full, mask_sliding= self._create_masks(s, x.device)
        
        # pass through the transformer blocks
        for i, block in enumerate(self.blocks):
            
            # check if its full or sliding attention
            if(self.layer_types[i] == "sliding_attention"):
                x= block(x, mask_sliding, self.sin_thetas_sliding, self.cos_thetas_sliding)
                
            else:
                x= block(x, mask_full, self.sin_thetas_full, self.cos_thetas_full)
                
        # final norm and output head
        x= self.final_norm(x)
        out= self.out_head(x.to(self.dtype)) # (b, s, vocab_size)
        
        return out
                
                
                
if __name__ == "__main__":
    
    model = GPT(GPT_CONFIG).to(DEVICE)
    
    # calc total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n~ Total number of parameters: {total_params:,}")
    
    
    # sample inputs
    x = torch.randint(1, 9, (2, GPT_CONFIG["context_length"]), dtype= torch.int32, device= DEVICE) # (B, S)
    y = torch.randint(1, 9, (2, GPT_CONFIG["context_length"]), dtype= torch.int32, device= DEVICE) # (B, S)
    
    
    logits = model(x)
    
    print("\n~ Input shape:", x.shape)
    print("~ Output shape:", y.shape)
    print("~ Logits shape:", logits.shape)
    


    # get the loss
    loss_func= nn.CrossEntropyLoss()
    
    ce_loss= loss_func(logits.to(torch.float32).flatten(0, 1), y.to(torch.long).view(-1))
    
    print(f"\n~ CE Loss: {ce_loss:.2f}")