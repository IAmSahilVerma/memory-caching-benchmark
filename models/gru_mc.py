import torch
import torch.nn as nn
from einops import rearrange

class ResidualMemeoryCache(nn.Module):
    """
    Residual Memeory Caching (RMC).
    Combines the current hidden state with a simple sum of all cached states.
    No learned parameters - purely additive.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(
        self,
        h_current: torch.Tensor,
        cache: list[torch.Tensor]
    ) -> torch.Tensor:
        # h_current : (batch, hidden_size)
        # cache     : list of (batch, hidden_size) tensors, one per past segment
        if not cache:
            return h_current
        stacked = torch.stack(cache, dim=1)     # (batch, num_segments, hidden_size)
        residual = stacked.sum(dim=1)           # (batch, hidden_size)
        return h_current + residual
    
class GatedResidualMemeoryCache(nn.Module):
    """
    Gated Residual Memeory Caching (GRMC).
    Each cached state gets an input-dependent scalar gate before summing.
    This lets the model learn which past segments are relevant to the current token.
    """
    def __init__(self, hidden_size: int, max_segments: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_segments = max_segments
        
        # Projects current hidden state to a gate score per cached segment
        self.gate_proj = nn.Linear(hidden_size, max_segments)
        self.sigmoid = nn.Sigmoid()
        
    def forward(
        self,
        h_current: torch.Tensor,
        cache: list[torch.Tensor]
    ) -> torch.Tensor:
        # h_current : (batch, hidden_size)
        # cache     : list of (batch, hidden_size) tensors
        if not cache:
            return h_current
        
        num_segments = len(cache)
        stacked = torch.stack(cache, dim=1) # (batch, num_segments, hidden_size)
        
        # Compute gates from current hidden state
        all_gates = self.gate_proj(h_current)   # (batch, max_segments)
        gates = self.sigmoid(
            all_gates[:, :num_segments] # (batch, num_segments)
        ).unsqueeze(-1)
        
        # Weighted sum of cached states
        weighted = (stacked * gates).sum(dim=1)
        return h_current + weighted
    
class GRUWithMemoryCache(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        segment_len: int = 64,
        mc_variant: str = "grmc", # "rmc" or "grmc"
        max_segments: int = 64
    ): 
        super().__init__()
        assert mc_variant in ("rmc", "grmc"), "mc_variant must be 'rmc' or 'grmc'"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.segment_len = segment_len
        self.mc_variant = mc_variant
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(
            input_size = d_model,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        
        # One cache module per GRU layer
        if mc_variant == "rmc":
            self.cache_modules = nn.ModuleList([
                ResidualMemeoryCache(hidden_size)
                for _ in range(num_layers)
            ])
        else:
            self.cache_modules = nn.ModuleList([
                GatedResidualMemeoryCache(hidden_size, max_segments)
                for _ in range(num_layers)
            ])
            
        self.head = nn.Linear(hidden_size, vocab_size)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)
                
    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor = None,
        cache: list[list[torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]:
        # x     : (batch, seq_len)
        # hidden: (num_layers, batch, hidden_size) or None
        # cache : list of num_layers lists, each holding past segment tensors
        
        batch_size, seq_len = x.shape
        device = x.device
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        if cache is None:
            cache = [[] for _ in range(self.num_layers)]
            
        emb = self.embedding(x) # (batch, seq_len, d_model)
        
        all_logits = []
        pos = 0
        
        while pos < seq_len:
            seg = emb[:, pos: pos + self.segment_len, :] # (batch, seg_len, d_model)
            
            # Apply cache to each layer's hidden state before processing segment
            hidden = self._apply_cache(hidden, cache)
            
            seg_out, hidden = self.gru(seg, hidden) # (batch, seg_len, hidden_size)
            
            # Save the last hidden state of this segment into cache
            # hidden[-1] is the top GRU layer; we cache all layers
            for layer_idx in range(self.num_layers):
                cache[layer_idx].append(
                    hidden[layer_idx].detach().clone() # (batch, hidden_size)
                )
                
            all_logits.append(seg_out)
            pos += self.segment_len
            
        out = torch.cat(all_logits, dim=1) # (batch, seq_len, hidden_size)
        out = self.dropout(out)
        logits = self.head(out) # (batch, seq_len, vocab_size)
        return logits, hidden, cache
        
    def _apply_cache(
        self,
        hidden: torch.Tensor,
        cache: list[list[torch.Tensor]]
    ) -> torch.Tensor:
        # hidden: (num_lyaers, batch, hidden_size)
        new_hidden_layers = []
        for layer_idx in range(self.num_layers):
            h_layer = hidden[layer_idx] # (batch, hidden_size)
            h_enhanced = self.cache_modules[layer_idx](h_layer, cache[layer_idx])
            new_hidden_layers.append(h_enhanced)
        return torch.stack(new_hidden_layers, dim=0) # (num_layers, batch, hidden_size)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
# Sanity Check
if __name__=="__main__":
    for variant in ("rmc", "grmc"):
        model = GRUWithMemoryCache(vocab_size=1000, mc_variant=variant)
        x = torch.randint(0, 1000, (4, 128))
        logits, hidden, cache = model(x)
        print(f"[{variant.upper()}] logits: {logits.shape} hidden: {hidden.shape}")
        print(f"[{variant.upper()}] params: {model.count_params():,}")