"""Entry point launching the Gradio UI."""

from __future__ import annotations

import gradio as gr

from .ui_components import create_interface


def launch() -> None:
    """Launch the UI."""
    demo = create_interface()
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    launch()
