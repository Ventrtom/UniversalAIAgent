## 🧠 Universal AI Agent

Multifunkční AI agent postavený na Langchainu, kombinující více nástrojů: JIRA retriever, webové vyhledávání a vektorové znalostní úloiště (RAG).

🌟 **Cíl**: Pomáhat s výzkumem, analýzou konkurence, správou roadmapy a návrhy v oblasti Product Managementu.

---

## 📌 Funkcionalita

Tento agent umožňuje:

* ✅ Vyhledávat informace na internetu (DuckDuckGo, Wikipedia – později Tavily).
* ✅ Ukládat výstupy do `.txt` souborů.
* ✅ Získávat tikety a popisy funkcí z JIRA API.
* ✅ Dotazovat se na interní znalosti uložené ve vektorové databázi (RAG).
* 🔄 Průběžně budovat znalostní základnu pomocí RAG, s možností zpětého ukládání výsledků dotazů.
* 🔄 Automaticky doplňovat znalosti z externích nástrojů do vektorového úloiště.
* 🔄 Detekovat duplicitní nebo zastaralé informace ve znalostní bázi a průběžně je čistit.

---

## 🧱 Architektura

```
            [User Prompt]
                 ↓
          [Langchain Agent]
                 ↓
 ┌────────────┌────────────┌────────────┍
 │  RAG Tool  │  Web Tool  │ JIRA Tool  │
 └────────────────────────────────────────────┘
      ↓              ↓           ↓
 Vector DB      Tavily Search   JIRA API
 (Chroma)         (planned)     (ready)
```

Agent vybírá vhodný nástroj na základě popisu (`Tool.description`) a relevance dotazu. Kombinuje výsledky, pokud je to potřeba.

---

## 📂 Struktura projektu

```
📆 universalagent/
├── .env                    # API klíče (OpenAI, JIRA)
├── .gitignore
├── main.py                # Hlavní běh agenta
├── tools.py               # Definice nástrojů (search, RAG, JIRA, save)
├── rag_vectorstore.py     # Indexace dokumentů do RAG
├── jira_retriever.py      # JIRA REST API přístup
├── sample_docs/           # Vstupní data pro testování
├── rag_chroma_db/         # Vektorové úloiště (ignored in git)
└── requirements.txt       # Python závislosti
```

---

## ⚙️ Instalace

```bash
pip install -r requirements.txt
```

---

## 🔐 Nastavení `.env`

```dotenv
# OpenAI API
OPENAI_API_KEY="..."

# JIRA přístup
JIRA_URL="https://your-domain.atlassian.net"
JIRA_USER="your-email@example.com"
JIRA_AUTH_TOKEN="your-jira-api-token"
JIRA_JQL="project = P4 ORDER BY created DESC"
JIRA_MAX_RESULTS=50
```

---

## 🚀 Spuštění agenta

```bash
python main.py
```

Agent se zeptá na dotaz a automaticky vybere nástroje. Např.:

```
What are our main competitors in the CMMS market?
```

---

## 🛠️ Integrované nástroje

| Nástroj             | Popis                                      |
| ------------------- | ------------------------------------------ |
| `searchWeb`         | DuckDuckGo (nahrazováno Tavily MCP)        |
| `Wikipedia`         | Shrnutí tématu                             |
| `rag_retriever`     | Vyhledávání ve znalostní bázi (Chroma RAG) |
| `jira_retriever`    | Přístup na JIRA (zatím samostatně)         |
| `save_text_to_file` | Uložení výstupu do souboru                 |

---

## 🔜 Plánované vylepšení

1. **Tavily MCP**: výměna DuckDuckGo za pokročilejší webové vyhledávání.
2. **JIRA Tool**: napojení `jira_retriever.py` jako Langchain Tool.
3. **Ukládání znalostí do RAG**: agent automaticky přidá nové poznatky do Chroma, pokud nejsou duplicitní.
4. **Detekce duplicity a expirovaných informací**: pomocí hashování textu, času, nebo vekt. vzdálenosti.
5. **Roadmap asistent**: kombinace JIRA výstupů + RAG + konkurence = návrhy co rozšířit, přidat nebo upřednostnit.
6. **IdeaRefiner**: AI doplňování a konkretizace nápadů v roadmapě.

---

## 🧪 Testovací dotazy

```
🔍 What are the challenges in predictive maintenance?
📌 Summarize recent JIRA issues related to Process Builder.
📚 What data sources does CMMS provide?
🤡 Suggest roadmap ideas based on current market competition.
```

---

## ✍️ Autor

**Tomáš Ventruba**
Produktový manažer a vývojář
Specializace: aplikace AI v průmyslovém softwaru

---

## ✅ Roadmapa

| Úloh                                    | Stav      |
| --------------------------------------- | --------- |
| ✅ Základní agent s RAG                  | Hotovo    |
| 🔄 Napojení JIRA do agenta              | Probíhá   |
| 🔄 Tavily web search                    | Plánováno |
| 🔄 Autonomní ukládání poznatků do RAG   | Plánováno |
| 🔄 Deduplication a cleaning ve vekt. DB | Plánováno |
| 🔄 Rozšiřování roadmapy pomocí AI       | Plánováno |

test