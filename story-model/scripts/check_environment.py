"""Print Python, torch, and accelerator availability for quick environment sanity checks."""

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")

    try:
        import torch
    except ImportError:
        print("torch: not installed")
        return

    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"MPS available: {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    main()
