# Zeno Workshop — Build Log

_Written like an engineering log. No marketing. Facts, decisions, results._

---

## Entry 001 — 2026-05-27 — Project Bootstrap

**What happened:**
Bootstrapped the Zeno Workshop repository from scratch.
This is the initial scaffolding session — no hardware tested yet, no LLM queries run yet.

**What was built:**

```
ZenoFinal/
├── zeno/
│   ├── __init__.py      — package marker
│   ├── config.py        — YAML config loader with defaults
│   ├── scanner.py       — project folder walker + context builder
│   ├── llm.py           — Ollama HTTP interface
│   ├── memory.py        — session notes + project summary persistence
│   ├── cards.py         — rich terminal card output
│   └── main.py          — CLI REPL
├── memory/
│   ├── PROJECT_MEMORY.md
│   ├── DECISIONS.md
│   ├── FAILURES.md
│   ├── BUILD_LOG.md     (this file)
│   ├── TASKS.md
│   ├── ROADMAP.md
│   └── TECH_STACK.md
├── test_project/        — synthetic ESP32 test project
├── config.yaml
├── requirements.txt
├── README.md
└── run.py
```

**Architecture decisions made:**
- CLI first, no GUI
- Ollama for local LLM (mistral default)
- Plain markdown for memory
- No agent framework
- rich library for terminal UI (cards, panels, tables)

**What is NOT done yet:**
- Not installed or tested on real hardware
- Ollama connection not verified
- No real Arduino/ESP32 project loaded yet
- No voice input
- No camera input
- No serial port reading

**Next action:**
Run `pip install -r requirements.txt` then `python run.py ./test_project` and verify it boots.

---

_Add new entries above this line._
