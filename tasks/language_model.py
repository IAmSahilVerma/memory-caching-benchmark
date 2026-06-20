import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset


class WikiTextLMDataset(Dataset):
    """
    Language Modelling task on WikiText-2, tokenized with GPT-2's BPE tokenizer.

    Unlike the synthetic tasks, this has no special marker/query tokens —
    it's standard next-token prediction. The raw text is tokenized once,
    concatenated into one long stream, then chopped into fixed-length
    chunks of `seq_len`. Each chunk's target is just the input shifted
    left by one position.

    This grounds the synthetic recall results in a real-world signal:
    does better recall actually translate to better language modelling?
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 256,
        config: str = "wikitext-2-raw-v1",
    ):
        super().__init__()
        self.seq_len = seq_len

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.pad_token = self.tokenizer.eos_token_id  # GPT-2 has no dedicated pad token
        self.total_vocab = self.tokenizer.vocab_size

        raw = load_dataset("Salesforce/wikitext", config, split=split)

        # Concatenate all non-empty lines into one long token stream
        text = "\n".join(t for t in raw["text"] if t.strip())
        token_ids = self.tokenizer.encode(text)

        self.tokens = torch.tensor(token_ids, dtype=torch.long)

        # Number of full (input, target) chunks of length seq_len we can extract
        # +1 because target is input shifted by one, so we need seq_len+1 tokens per chunk
        self.num_chunks = (len(self.tokens) - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end   = start + self.seq_len

        input_seq  = self.tokens[start: end]
        target_seq = self.tokens[start + 1: end + 1]

        return input_seq, target_seq
    
if __name__ == "__main__":
    ds = WikiTextLMDataset(split="validation", seq_len=32)  # validation is smaller, faster to test
    print(f"Total vocab   : {ds.total_vocab}")
    print(f"Total tokens  : {len(ds.tokens)}")
    print(f"Num chunks    : {len(ds)}")

    x, y = ds[0]
    print(f"Input  shape  : {x.shape}")
    print(f"Target shape  : {y.shape}")
    print(f"Input  tokens : {x.tolist()}")
    print(f"Target tokens : {y.tolist()}")

    # Verify the shift-by-one relationship: target[i] should equal input[i+1]
    # for all but the last position, where target[-1] is the *next* unseen token
    shift_check = torch.equal(y[:-1], x[1:])
    print(f"Shift-by-one consistency (target[:-1] == input[1:]) : {shift_check}")

    # Decode a snippet to sanity-check it reads like real English
    decoded = ds.tokenizer.decode(x.tolist())
    print(f"Decoded input : {decoded!r}")