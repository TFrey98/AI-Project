"""Training entry point: reads a config, trains a model, and writes checkpoints."""

import argparse
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import save_checkpoint
from story_model.data import CharTokenizer, get_batch, load_text, train_test_split
from story_model.models.bigram import BigramLanguageModel


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(config_path: str) -> None:
    config = yaml.safe_load(Path(config_path).read_text())

    torch.manual_seed(config["train"]["seed"])
    device = resolve_device(config["train"]["device"])

    text = load_text(config["data"]["path"])
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = train_test_split(data, config["data"]["train_split"])

    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])

    block_size = config["data"]["block_size"]
    batch_size = config["data"]["batch_size"]

    for step in range(config["train"]["max_steps"]):
        if step % config["train"]["eval_interval"] == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                block_size,
                batch_size,
                config["train"]["eval_iters"],
                device,
            )
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if step > 0 and step % config["checkpoint"]["save_interval"] == 0:
            save_checkpoint(
                Path(config["checkpoint"]["dir"]) / f"step_{step}.pt",
                model,
                optimizer,
                step,
            )

        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(Path(config["checkpoint"]["dir"]) / "final.pt", model, optimizer, config["train"]["max_steps"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a training config YAML file.")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
