import torch
import torch.nn as nn

class GRUBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(
            input_size = d_model,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, vocab_size)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)
                
    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x : (batch, seq_len)
        # hidden : (num_layers, batch, hidden_size) or None
        out = self.embedding(x)     # (batch, seq_len, d_model)
        out, hidden = self.gru(out, hidden) # (batch, seq_len, hidden_size)
        out = self.dropout(out)
        logits = self.head(out)
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
# Sanity Check
if __name__ == "__main__":
    model = GRUBaseline(vocab_size=1000)
    x = torch.randint(0, 1000, (4, 128))
    logits, hidden = model(x)
    print(f"Logits shape    :   {logits.shape}")
    print(f"Hidden shape    :   {hidden.shape}")
    print(f"Params          :   {model.count_params():,}")