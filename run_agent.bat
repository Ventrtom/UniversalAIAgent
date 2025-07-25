@echo off
REM === PŘEPNE SE DO SLOŽKY PROJEKTU ===
cd /d "C:\Users\tomas\OneDrive\Python\ProductManagemensAssistant\UniversalAIAgent"

REM === BUĎ (A) AKTIVUJI VENV A VOLÁM python ===
call "venv\Scripts\activate.bat"
python run.py

REM === ALTERNATIVA (B) BEZ AKTIVACE ===
:: "venv\Scripts\python.exe" run.py

REM === CHCEŠ‑LI NECHAT OKNO OTEVŘENÉ PRO LOGY, ODKOMENTUJ: ===
:: pause
