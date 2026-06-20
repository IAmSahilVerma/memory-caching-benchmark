import torch
from torch.utils.data import Dataset


class SelectiveCopyDataset(Dataset):
    """
    Selective Copy task (Gu & Dao, 2023 / Mamba paper formulation).

    A sequence of random tokens contains a small number of 'marked' tokens,
    scattered at random positions, encoded in a separate ID range (value +
    vocab_size) so the model can identify them directly from the token ID.

    After a COPY marker, the model must reproduce the marked values in the
    order they occurred, AUTOREGRESSIVELY: each recalled value is fed back
    in as input for predicting the next one. This is what makes the task
    learnable — at each recall step, the model's hidden state has just
    incorporated which value it produced last, distinguishing "recall the
    2nd value" from "recall the 1st value." A single-shot parallel query
    (predicting all values at once from identical query tokens) gives the
    model no way to know which slot is which, so this sequential design is
    essential, not a simplification.

    Example (vocab_size=10):
        body   : [3, 17, 5, 8, 12, 1]      <- 17, 12 are marked (7+10, 2+10)
        seq    : [3, 17, 5, 8, 12, 1, COPY, 7, 2]
        target : [pad, pad, pad, pad, pad, pad, 7, 2, pad]
                  (target at position i = the token that SHOULD appear at i+1;
                   position of COPY predicts the first recalled value, the
                   position holding the first recalled value predicts the
                   second, and so on)
    """

    def __init__(
        self,
        num_samples: int,
        seq_len: int = 128,
        num_marked: int = 8,
        vocab_size: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.num_marked = num_marked
        self.vocab_size = vocab_size

        # Token ID layout:
        #   [0, vocab_size)             -> normal tokens
        #   [vocab_size, 2*vocab_size)  -> marked tokens (value = id - vocab_size)
        #   2*vocab_size                -> COPY marker token
        #   2*vocab_size + 1            -> pad token (ignored in loss)
        self.copy_token  = 2 * vocab_size
        self.pad_token   = 2 * vocab_size + 1
        self.total_vocab = 2 * vocab_size + 2

        # body_len + 1 (COPY token) + num_marked (recalled values) = seq_len
        self.body_len = seq_len - 1 - num_marked
        self.seq_len  = seq_len

        self.generator = torch.Generator().manual_seed(seed)
        self.data = self._generate(num_samples)

    def _generate(self, num_samples: int) -> list[dict]:
        samples = []

        for _ in range(num_samples):
            body = torch.randint(
                0, self.vocab_size, (self.body_len,), generator=self.generator
            )

            mark_positions = torch.randperm(
                self.body_len, generator=self.generator
            )[: self.num_marked]
            mark_positions, _ = torch.sort(mark_positions)

            marked_values = body[mark_positions].clone()   # order = appearance order

            input_body = body.clone()
            input_body[mark_positions] += self.vocab_size

            # Full input: body, COPY, then the recalled values themselves
            # (model sees its own correct outputs during training — standard
            # teacher forcing, same as language modelling)
            input_seq = torch.cat([
                input_body,
                torch.tensor([self.copy_token], dtype=torch.long),
                marked_values,
            ])

            # Target: shift-by-one, just like language modelling. Pad
            # everywhere except where the next token is a value to recall.
            target = torch.full(
                (self.seq_len,), self.pad_token, dtype=torch.long
            )
            # Position of COPY token predicts marked_values[0]
            copy_pos = self.body_len
            target[copy_pos: copy_pos + self.num_marked] = marked_values

            samples.append({"input": input_seq, "target": target})

        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        return item["input"], item["target"]
    
if __name__ == "__main__":
    ds = SelectiveCopyDataset(num_samples=4, seq_len=32, num_marked=4, vocab_size=10)
    x, y = ds[0]
    print(f"Input  : {x.tolist()}")
    print(f"Target : {y.tolist()}")
    print(f"Total vocab : {ds.total_vocab}")
    print(f"Body len : {ds.body_len}")

    body = x[:ds.body_len]
    marked_mask = body >= ds.vocab_size
    marked_values = (body[marked_mask] - ds.vocab_size).tolist()
    print(f"Marked values (order of appearance) : {marked_values}")

    copy_pos = ds.body_len
    recalled_targets = y[copy_pos: copy_pos + ds.num_marked].tolist()
    print(f"Target values at COPY+ positions     : {recalled_targets}")
    print(f"Match: {marked_values == recalled_targets}")