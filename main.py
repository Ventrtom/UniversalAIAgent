"""Compatibility wrapper for the CLI entry point."""
from __future__ import annotations

from agent import agent_executor
from cli.main import main

__all__ = ["agent_executor", "main"]

if __name__ == "__main__":
    main()
