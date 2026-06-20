import json
import torch
from torch.utils.data import DataLoader
import os

from models.transformer import SmallTransformer
from models.gru import GRUBaseline
from models.gru_mc import GRUWithMemoryCache
from tasks.selective_copy import SelectiveCopyDataset
from tasks.associative_recall import AssociativeRecallDataset
from tasks.language_model import WikiTextLMDataset
from train import train_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name: str, vocab_size: int, task_name: str):
    """Factory: returns a freshly initialized model for a given vocab size,
    with GRU hidden_size chosen per-task so total parameter count is matched
    to the Transformer's (~810K on synthetic tasks, ~13.7M on language_model).
    Without this, the Transformer has ~3.8x more parameters than the GRU
    variants on the synthetic tasks, confounding any architecture comparison."""
    gru_hidden = 274 if task_name in ("selective_copy", "associative_recall") else 140

    if model_name == "transformer":
        return SmallTransformer(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=4)
    elif model_name == "gru":
        return GRUBaseline(vocab_size=vocab_size, d_model=128, hidden_size=gru_hidden, num_layers=2)
    elif model_name == "gru_rmc":
        return GRUWithMemoryCache(vocab_size=vocab_size, d_model=128, hidden_size=gru_hidden,
                                num_layers=2, mc_variant="rmc", segment_len=32)
    elif model_name == "gru_grmc":
        return GRUWithMemoryCache(vocab_size=vocab_size, d_model=128, hidden_size=gru_hidden,
                                num_layers=2, mc_variant="grmc", segment_len=32)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def build_task(task_name: str):
    """
    Returns (train_dataset, val_dataset, pad_token).
    Train/val split for synthetic tasks is just two independently-seeded
    generations; for LM it's the dataset's real train/validation splits.
    """
    if task_name == "selective_copy":
        train_ds = SelectiveCopyDataset(num_samples=2000, seq_len=128, num_marked=8, vocab_size=32, seed=1)
        val_ds   = SelectiveCopyDataset(num_samples=400,  seq_len=128, num_marked=8, vocab_size=32, seed=2)
        return train_ds, val_ds, train_ds.pad_token, train_ds.total_vocab

    elif task_name == "associative_recall":
        train_ds = AssociativeRecallDataset(num_samples=2000, num_pairs=16, key_vocab_size=32, value_vocab_size=32, seed=1)
        val_ds   = AssociativeRecallDataset(num_samples=400,  num_pairs=16, key_vocab_size=32, value_vocab_size=32, seed=2)
        return train_ds, val_ds, train_ds.pad_token, train_ds.total_vocab

    elif task_name == "language_model":
        train_ds = WikiTextLMDataset(split="train", seq_len=128)
        val_ds   = WikiTextLMDataset(split="validation", seq_len=128)
        return train_ds, val_ds, train_ds.pad_token, train_ds.total_vocab

    else:
        raise ValueError(f"Unknown task_name: {task_name}")


def run_all_experiments(
    model_names: list[str] = ("transformer", "gru", "gru_rmc", "gru_grmc"),
    task_names: list[str] = ("selective_copy", "associative_recall", "language_model"),
    num_epochs: int = 5,
    batch_size: int = 32,
    output_path: str = "results/all_results.json",
):
    all_results = {}

    for task_name in task_names:
        print(f"\n{'='*60}\nTASK: {task_name}\n{'='*60}")
        train_ds, val_ds, pad_token, vocab_size = build_task(task_name)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        all_results[task_name] = {}

        for model_name in model_names:
            print(f"\n--- {task_name} / {model_name} ---")
            model = build_model(model_name, vocab_size, task_name)

            # GRU variants need a higher LR to compensate for weaker gradient signal
            # through the recurrent stack vs. the Transformer's attention pathways
            # lr = 1e-3 if model_name.startswith("gru") else 3e-4

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                pad_token=pad_token,
                num_epochs=num_epochs,
                lr=3e-4,
                device=DEVICE,
                model_name=f"{task_name}/{model_name}",
            )

            all_results[task_name][model_name] = history

            # Save incrementally so a crash partway through doesn't lose everything
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nAll experiments complete. Results saved to {output_path}")
    return all_results

if __name__ == "__main__":
    if not os.path.isfile("results/all_results.json"):
        print("Running Full Sweep")
        print("--------------------------------")
        run_all_experiments(
            model_names=["transformer", "gru", "gru_rmc", "gru_grmc"],
            task_names=["selective_copy", "associative_recall", "language_model"],
            num_epochs=40,
            output_path="results/all_results.json",
        )
    elif not os.path.isfile("results/run_b_gru_selective_copy_extended.json"):
        print("Running GRU on Selective Copy")
        print("--------------------------------")
        run_all_experiments(
        model_names=["gru", "gru_rmc", "gru_grmc"],
        task_names=["selective_copy"],
        num_epochs=80,
        output_path="results/run_b_gru_selective_copy_extended.json",
    )