import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                      # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)  — token ids
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        out = self.embedding(x) * math.sqrt(self.d_model)
        out = self.pos_enc(out)
        out = self.transformer(out, mask=mask, is_causal=True)
        return self.head(out)           # (batch, seq_len, vocab_size)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == "__main__":
    model = SmallTransformer(vocab_size=1000)
    x = torch.randint(0, 1000, (4, 128))
    out = model(x)
    print(f"Output shape : {out.shape}")       # expect (4, 128, 1000)
    print(f"Params       : {model.count_params():,}")  # expect ~5-7M