"""Backward-compatible entry point. Use main.py for the actual application."""
from main import main

if __name__ == "__main__":
    raise SystemExit(main())
