# Zeno Workshop — Build Log

_Written like an engineering log. No marketing. Facts, decisions, results._

---

## Entry 004 — 2026-05-27 — Real Project Q&A Check

**What happened:**
Ran Zeno against the actual Zeno workspace instead of the synthetic ESP32 test project and asked three concrete questions.

**Questions asked:**
- What does this project do right now?
- How do I launch the HUD prototype?
- Where are structured Q&A captures and session closeouts saved?

**Observed result:**
- HUD launch question: correct after fallback fix
- Memory location question: correct
- Project-description question: acceptable, but still drifts into the embedded `test_project` instead of staying tightly focused on Zeno itself

**Bug found and fixed:**
- Empty LLM responses were treated as valid answers
- Added `[EMPTY RESPONSE]` handling in `zeno/llm.py`
- Added deterministic fallback answers for HUD launch and memory-file location questions in `zeno/analyzer.py`

**Current score:**
2/3 answers clean
1/3 answers usable but not sharp enough

**Next action:**
Improve project-description answers so they prioritize the loaded project itself over example subprojects, then rerun the same 3-question check.

---

## Entry 003 — 2026-05-27 — Memory Capture and HUD Prototype

**What happened:**
Implemented the next two priorities in order: better memory capture, then a minimal HUD prototype.

**What was added:**
- `:capture` — saves the last question and answer to `memory/INTERACTIONS.md`
- `:closeout` — saves a structured session closeout to `memory/SESSION_CLOSEOUTS.md`
- `zeno/hud.py` — Tkinter-based desktop HUD prototype
- `python run.py --hud <project_path>` — alternate entrypoint for the HUD

**HUD card set in this prototype:**
- Project status
- Detected issues
- Last answer
- Ollama status embedded in project card

**What was tested:**
- Python syntax compile including `zeno/hud.py`
- Real CLI run verifying `:capture` and `:closeout`
- File creation checks for `INTERACTIONS.md` and `SESSION_CLOSEOUTS.md`
- Functional smoke test of `build_hud_state()` against `test_project`

**Observed result:**
The assistant can now persist structured interactions and session closeouts, and the HUD state builder produces live project/issue/answer data from the same core logic as the CLI.

**What still needs proof:**
- Manual usability check of the Tkinter HUD window during an actual workflow
- Whether the HUD should remain a plain window or move toward a transparent overlay
- Whether the current cards are the right ones for live debugging

**Next action:**
Open the HUD window against a real project folder and decide whether to push toward overlay behavior or richer debugging cards.

---

## Entry 002 — 2026-05-27 — Demo 001 Made Usable

**What happened:**
Closed the largest reliability gap in Demo 001.
The original scaffold depended too heavily on the LLM for basic project inspection, which meant the assistant became weak whenever the model was slow, wrong, or unavailable.

**What was added:**
- `zeno/analyzer.py` — deterministic local analysis layer
- `:localsummary` — project summary without using the LLM
- `:issues` — static issue scan for obvious faults
- `:logs` — extracted error lines from `.log` files
- `:inspect <file>` — line-numbered file excerpt card
- `:save` — save the last response directly into session memory
- LLM fallback path: if Ollama is unavailable or returns an error, Zeno now falls back to local analysis for common engineering questions

**What was tested:**
- Python syntax compile for all source files
- Analyzer smoke test against `test_project`
- Real CLI run with scripted commands:
	- `:localsummary`
	- `:issues`
	- `:inspect errors.log`
	- `:quit`

**Observed result:**
The CLI successfully loaded the project, displayed the new commands, produced a local issue summary, extracted the watchdog reset from `errors.log`, and rendered a line-numbered file inspection card.

**What still needs proof:**
- Q&A quality on a real project folder, not just the synthetic test project
- `:summarize` output quality with the current model (`malixator/ZenoV1`)
- Session memory write verification through `:save` in an interactive run

**Next action:**
Run Demo 001 on one real project folder and verify 3 answers against known facts before adding Phase 2 features.

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
