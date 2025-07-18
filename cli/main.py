# cli/main.py
from __future__ import annotations

from agent import handle_query
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import ResearchResponse

from agent import ResearchResponse as _RR

def main() -> None:
    """Launch interactive CLI until user exits."""
    print("\nUniversal AI Agent (Productoo) — napište 'exit' pro ukončení.\n")
    while True:
        try:
            user_query = input("Vy: ").strip()
            if user_query.lower() in {"exit", "quit", "bye"}:
                print("Asistent: Rád jsem pomohl! Mějte se.")
                break

            raw = handle_query(user_query)

            try:
                parsed: 'ResearchResponse' = _RR.parse_raw(raw)
                print("\nAsistent:",
                      textwrap.fill(parsed.answer, 100), "\n")
            except Exception:
                print("\nAsistent:", raw)

        except KeyboardInterrupt:
            print("\nAsistent: Končím. Mějte se.")
            break


if __name__ == "__main__":
    main()
