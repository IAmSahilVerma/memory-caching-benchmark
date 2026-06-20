import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_model_outputs(model: nn.Module, x: torch.Tensor):
    """
    Normalizes the forward-pass signature across all model types so the
    training loop doesn't need to know which model it's dealing with.

    - SmallTransformer.forward(x)              -> logits
    - GRUBaseline.forward(x)                   -> (logits, hidden)
    - GRUWithMemoryCache.forward(x)             -> (logits, hidden, cache)
    """
    out = model(x)
    if isinstance(out, tuple):
        return out[0]          # logits is always first
    return out


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = get_model_outputs(model, x)        # (batch, seq_len, vocab_size)

        # CrossEntropyLoss expects (batch * seq_len, vocab_size) vs (batch * seq_len,)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_token: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total_predicted = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        logits = get_model_outputs(model, x)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        total_loss += loss.item()
        num_batches += 1

        # Accuracy only over non-pad positions (the positions that matter)
        preds = logits.argmax(dim=-1)
        mask  = (y != pad_token)
        correct += (preds[mask] == y[mask]).sum().item()
        total_predicted += mask.sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / total_predicted if total_predicted > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pad_token: int,
    num_epochs: int = 10,
    lr: float = 3e-4,
    device: torch.device = None,
    model_name: str = "model",
) -> dict:
    """
    Full training run for one model. Returns a history dict suitable for
    dumping straight to JSON for the report stage.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    history = {
        "model_name": model_name,
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time_sec": [],
    }

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, pad_token)

        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["epoch_time_sec"].append(elapsed)

        print(
            f"[{model_name}] epoch {epoch:>2}/{num_epochs} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | "
            f"val_acc {val_metrics['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

    return history