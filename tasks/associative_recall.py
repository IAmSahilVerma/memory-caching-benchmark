import torch
from torch.utils.data import Dataset


class AssociativeRecallDataset(Dataset):
    """
    Associative Recall task.

    A sequence of key-value pairs is presented early on. Later, a single
    query key appears, and the model must output the value that was
    originally paired with it.

    This is the hardest recall task for fixed-memory models: unlike
    Selective Copy, the model can't just "remember the marked tokens" —
    it has to retain an actual key->value mapping and do a lookup, which
    stresses associative (not just positional) memory.

    Example (vocab_size=10, num_pairs=4):
        pairs    : [k0, v0, k1, v1, k2, v2, k3, v3]
                 = [3,  7,  1,  9,  5,  2,  8,  4]
        query    : [k2]               -> looking for the value paired with key 5
        target   : [..., 2]           -> only the final position has a real target
    """

    def __init__(
        self,
        num_samples: int,
        num_pairs: int = 16,
        key_vocab_size: int = 32,
        value_vocab_size: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.num_pairs        = num_pairs
        self.key_vocab_size   = key_vocab_size
        self.value_vocab_size = value_vocab_size

        # Token ID layout:
        #   [0, key_vocab_size)                                  -> key tokens
        #   [key_vocab_size, key_vocab_size + value_vocab_size)  -> value tokens
        #   next id                                              -> query marker token
        #   next id                                               -> pad token (ignored in loss)
        self.value_offset  = key_vocab_size
        self.query_token   = key_vocab_size + value_vocab_size
        self.pad_token      = key_vocab_size + value_vocab_size + 1
        self.total_vocab    = key_vocab_size + value_vocab_size + 2

        # seq layout: [k0, v0, k1, v1, ..., k_{n-1}, v_{n-1}, QUERY, query_key]
        self.seq_len = 2 * num_pairs + 2

        self.generator = torch.Generator().manual_seed(seed)
        self.data = self._generate(num_samples)

    def _generate(self, num_samples: int) -> list[dict]:
        samples = []

        for _ in range(num_samples):
            # Sample distinct keys so each key maps to exactly one value
            keys = torch.randperm(
                self.key_vocab_size, generator=self.generator
            )[: self.num_pairs]

            values = torch.randint(
                0, self.value_vocab_size, (self.num_pairs,), generator=self.generator
            )

            # Interleave keys and values: k0, v0, k1, v1, ...
            pairs = torch.empty(2 * self.num_pairs, dtype=torch.long)
            pairs[0::2] = keys
            pairs[1::2] = values + self.value_offset

            # Pick which pair to query
            query_idx = torch.randint(
                0, self.num_pairs, (1,), generator=self.generator
            ).item()
            query_key = keys[query_idx]
            answer_value = values[query_idx]

            # Final sequence: pairs + [QUERY token, query_key]
            input_seq = torch.cat([
                pairs,
                torch.tensor([self.query_token, query_key], dtype=torch.long),
            ])

            # Target: pad everywhere except the very last position
            target = torch.full(
                (self.seq_len,), self.pad_token, dtype=torch.long
            )
            target[-1] = answer_value + self.value_offset

            samples.append({"input": input_seq, "target": target})

        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        return item["input"], item["target"]

if __name__ == "__main__":
    ds = AssociativeRecallDataset(num_samples=4, num_pairs=4, key_vocab_size=10, value_vocab_size=10)
    x, y = ds[0]
    print(f"Input  : {x.tolist()}")
    print(f"Target : {y.tolist()}")
    print(f"Total vocab : {ds.total_vocab}")

    # query token marker is second-to-last, the actual query key is last
    query_token = x[-2].item()
    query_key   = x[-1].item()
    pairs = x[:-2]
    keys = pairs[0::2]
    values = pairs[1::2]

    match_idx = (keys == query_key).nonzero().item()
    expected_value = values[match_idx].item()

    print(f"Query token (marker) : {query_token}")
    print(f"Query key            : {query_key}")
    print(f"Expected value (shifted) : {expected_value}")
    print(f"Target value (shifted)   : {y[-1].item()}")