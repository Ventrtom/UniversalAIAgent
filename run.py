"""Compatibility wrapper for the Gradio UI entry point."""
from cli.ui import launch
import os
from agent import start_background_jobs

if __name__ == "__main__":
    start_background_jobs(
        {
            "snapshot_interval": int(os.getenv("CHAT_SNAPSHOT_INTERVAL", 600)),
            "snapshot_path": os.getenv("CHAT_SNAPSHOT_FILE", "data/chat_snapshot.jsonl"),
            "mask_pii": os.getenv("CHAT_SNAPSHOT_MASK_PII", "false").lower() in {"1", "true", "yes"},
        }
    )
    launch()
