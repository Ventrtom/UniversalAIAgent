# cli/main2.py
from __future__ import annotations
from agent.core2 import handle_query   # ← důležité

BANNER = "\nUniversal AI Agent • *core2* (LangGraph) — napište 'exit' pro ukončení.\n"

def main() -> None:
    print(BANNER)
    while True:
        try:
            q = input("Vy: ").strip()
            if q.lower() in {"exit", "quit", "bye"}:
                print("Asistent: Rád jsem pomohl! Mějte se.")
                break
            print("\nAsistent:", handle_query(q), "\n")
        except KeyboardInterrupt:
            print("\nAsistent: Končím. Mějte se.")
            break

if __name__ == "__main__":
    main()
