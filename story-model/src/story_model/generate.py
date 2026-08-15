"""Sample text from a trained checkpoint."""

import argparse
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import load_checkpoint
from story_model.data import CharTokenizer, load_text
from story_model.models.bigram import BigramLanguageModel


def generate(config_path: str, checkpoint_path: str, max_new_tokens: int) -> str:
    config = yaml.safe_load(Path(config_path).read_text())

    text = load_text(config["data"]["path"])
    tokenizer = CharTokenizer.from_text(text)

    model = BigramLanguageModel(tokenizer.vocab_size)
    load_checkpoint(checkpoint_path, model, map_location="cpu")
    model.eval()

    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    return tokenizer.decode(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the training config YAML file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    print(generate(args.config, args.checkpoint, args.max_new_tokens))


if __name__ == "__main__":
    main()
