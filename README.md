# Zeno Workshop

A room-built AI assistant for makers and embedded engineers.

Reads your project folder, understands your code, diagnoses errors, and answers engineering questions — running entirely locally, no internet required.

**Current phase: Demo 001 — Project-Aware CLI**

---

## What it does right now

- Load any project folder (Arduino, ESP32, Python, C/C++)
- Build a full context from your code, README, and log files
- Answer engineering questions about the project via local LLM
- Fall back to deterministic local analysis when the LLM is unavailable or unhelpful
- Identify errors in log files and extract crash lines
- Inspect specific files with line-numbered excerpts
- Run a local issue scan for obvious embedded/software faults
- Save session notes, structured Q&A captures, and session closeouts to `./memory/`
- Display everything as clean terminal cards
- Launch a minimal desktop HUD prototype with project status, issues, and last answer cards

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

### 2. Install Ollama

Download from https://ollama.com and install.

Then start the server:
```bash
ollama serve
```

### 3. Pull a model

```bash
ollama pull malixator/ZenoV1
```

Or use any installed local model and set it in `config.yaml`.

If you change the model, update `config.yaml`:
```yaml
llm:
  model: malixator/ZenoV1
```

### 4. Run Zeno

```bash
python run.py                        # prompts for project path
python run.py ./test_project         # load the included test project
python run.py /path/to/your/project  # load any real project
python run.py --hud ./test_project   # desktop HUD prototype
```

---

## Commands

| Command | Description |
|---------|-------------|
| `<question>` | Ask anything about the loaded project |
| `:load <path>` | Load a different project folder |
| `:tree` | Show project file tree |
| `:files` | List all loaded files |
| `:inspect <file>` | Show a file excerpt with line numbers |
| `:issues` | Run deterministic issue detection |
| `:logs` | Show extracted error lines from `.log` files |
| `:localsummary` | Build a local summary without using the LLM |
| `:summarize` | Ask Zeno to summarize the project |
| `:note <text>` | Save a note to memory/SESSION_NOTES.md |
| `:save` | Save the last response to SESSION_NOTES.md |
| `:capture` | Save the last Q&A to memory/INTERACTIONS.md |
| `:closeout a|b|c|d|e|f` | Save a structured session closeout |
| `:status` | Check Ollama connection |
| `:clear` | Clear screen |
| `:help` | Show command reference |
| `:quit` | Exit |

---

## Example queries

```
What does this project do?
What libraries does this use?
Why is the MPU6050 not responding?
What is causing the error in errors.log?
Why is the firmware crashing after BLE disconnect?
How should I wire the SDA/SCL pins for this board?
What is the next thing I should implement?
Is there anything dangerous in this code?
```

---

## Configuration

Edit `config.yaml`:

```yaml
llm:
  model: malixator/ZenoV1 # change to any model you already have in Ollama
  timeout: 90             # increase on slow CPU hardware

project:
  max_file_size_kb: 150   # skip very large files
```

---

## Project structure

```
ZenoFinal/
├── zeno/            Python package — scanner, LLM, memory, UI
├── memory/          Persistent memory files (markdown)
├── test_project/    Synthetic ESP32 project for testing
├── config.yaml      Configuration
├── requirements.txt Python dependencies
└── run.py           Entry point
```

---

## Memory files

Zeno keeps all notes in `./memory/`:

| File | Purpose |
|------|---------|
| `PROJECT_MEMORY.md` | Project state, goals, stack |
| `DECISIONS.md` | Architecture decision log |
| `FAILURES.md` | Failure and lessons log |
| `BUILD_LOG.md` | Engineering build log |
| `TASKS.md` | Task list (today / week / later) |
| `ROADMAP.md` | Phase-by-phase roadmap |
| `TECH_STACK.md` | Tools, models, protocols |
| `SESSION_NOTES.md` | Notes saved during sessions |
| `INTERACTIONS.md` | Structured Q&A captures |
| `SESSION_CLOSEOUTS.md` | End-of-session structured closeouts |
| `PROJECT_SUMMARIES.md` | Auto-generated project summaries |

---

## Roadmap summary

| Phase | Goal |
|-------|------|
| 1 (now) | Project-aware CLI assistant |
| 2 | Serial/log live debug + voice input |
| 3 | Desktop HUD overlay + camera |
| 4 | ESP32 gesture glove control |
| 5 | AR glasses HUD bridge |
| 6 | Public launch package |

Full detail: [memory/ROADMAP.md](memory/ROADMAP.md)

---

## Hardware target

- Windows / Linux / Raspberry Pi (local Ollama)
- ESP32 + MPU6050 gesture glove (Phase 4)
- Monocular AR glasses (Phase 5)

---

## Why this exists

This is not a product. It is a maker/engineer's personal tool built from scraps.
The goal is to document the construction publicly, accumulate proof of work, and build something genuinely useful in a room.

See [memory/BUILD_LOG.md](memory/BUILD_LOG.md) for the full engineering log.

---

## License

MIT
