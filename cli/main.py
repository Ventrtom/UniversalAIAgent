# cli/main.py
from __future__ import annotations

from agent import handle_query, parser, ResearchResponse


def main() -> None:
    """Launch interactive CLI until user exits."""
    print("\nUniversal AI Agent (Productoo) — napište 'exit' pro ukončení.\n")
    while True:
        try:
            user_query = input("Vy: ").strip()
            if user_query.lower() in {"exit", "quit", "bye"}:
                print("Asistent: Rád jsem pomohl! Mějte se.")
                break

            answer = handle_query(user_query)

            try:
                structured: ResearchResponse = parser.parse(answer)
                print(f"\nAsistent: {structured.summary}\n")
                if structured.sources:
                    print("Zdroj(e):", ", ".join(structured.sources))
            except Exception:
                print("\nAsistent:", answer)

        except KeyboardInterrupt:
            print("\nAsistent: Končím. Mějte se.")
            break


if __name__ == "__main__":
    main()
