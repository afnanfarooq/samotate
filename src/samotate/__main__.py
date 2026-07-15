"""Allow running the CLI via ``python -m samotate``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
