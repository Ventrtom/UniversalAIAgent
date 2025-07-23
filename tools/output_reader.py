"""LangChain tools for listing and reading files from the shared folder."""
from __future__ import annotations

from langchain.tools import Tool
from services.output_reader import list_output_files, read_output_file


def _list() -> str:
    return list_output_files()


def _read(name: str) -> str:
    return read_output_file(name)


list_output_files_tool = Tool(
    name="list_output_files",
    func=_list,
    description=(
        "Return names of files stored under ./files/ with extensions txt, md, csv or pdf."
    ),
    handle_tool_error=True,
)

read_output_file_tool = Tool(
    name="read_output_file",
    func=_read,
    description=(
        "Read the content of a specific file from ./files/. The argument must be an existing file name."
    ),
    handle_tool_error=True,
)

__all__ = ["list_output_files_tool", "read_output_file_tool"]
