# Zeno Workshop — Decision Log

---

## 2026-05-27 — Bootstrap Architecture

**Decision:** Start Demo 001 as a pure CLI tool (no GUI, no voice, no camera).

**Why:**
A working CLI that actually answers questions about a project is useful immediately.
Every layer added on top of a broken foundation makes debugging harder.
CLI first → HUD second → voice/camera third.

**Alternatives rejected:**
- GUI from the start (Tkinter/PyQt): Too much setup friction. Adds zero value until the LLM integration works.
- Web UI (Flask/FastAPI): Overkill for local personal tool in Week 1.
- Agent framework (LangChain/AutoGen): Massive dependency tree, obscures what's actually happening, hard to debug on limited hardware.
- Full AR pipeline first: Completely wrong order. You cannot debug AR without first having a working assistant.

**Risk:** CLI is less impressive in demos.
**Mitigation:** Demo 001 video will show terminal output directly. Rich cards look clean on screen.

**Revisit:** After Demo 001 is working and committed.

---

## 2026-05-27 — LLM Backend: Ollama

**Decision:** Use Ollama for local LLM inference. Default model: mistral.

**Why:**
- Completely offline. No API keys. No cost. No data leaving the machine.
- Ollama is the most friction-free local LLM setup available.
- mistral-7B runs acceptably on CPU-only systems (slow but usable).
- Model is swappable via config.yaml — no code changes needed.

**Alternatives rejected:**
- OpenAI API: Costs money. Sends project code to external servers. Bad fit for a "made from scraps" tool.
- llama.cpp direct: More control but requires manual model management and GGUF format knowledge.
- Hugging Face transformers: Heavy dependencies, slow setup, not beginner-hostile enough to justify.
- GPT4All: Less maintained, smaller community, fewer models.

**Risk:** mistral is slow on CPU. Responses can take 30-90s without GPU.
**Mitigation:** config.yaml timeout is 90s. User can switch to smaller models (phi3-mini, tinyllama).

**Revisit:** When hardware improves or Raspberry Pi becomes the inference server.

---

## 2026-05-27 — Memory Format: Markdown Files

**Decision:** All persistent memory is plain markdown files in ./memory/

**Why:**
- Human-readable. No database to maintain. No migration scripts.
- Can be read and edited manually at any time.
- Git-tracked naturally — commit history is the audit trail.
- Easy to grep, search, and reference from LLM context.

**Alternatives rejected:**
- SQLite: Overkill for this scale. Harder to read manually.
- JSON files: Less readable for human review and editing.
- External vector database (Chroma, Qdrant): Phase 3+. Not justified for Demo 001.

**Risk:** Markdown files get large and unstructured over time.
**Mitigation:** Each file has a defined format. Will add rotation/archiving when files exceed ~500 lines.

---

## 2026-05-27 — No Agent Framework

**Decision:** Do not use LangChain, AutoGen, CrewAI, or similar frameworks for Demo 001.

**Why:**
These frameworks obscure control flow, have large dependency trees, break frequently with model updates,
and are difficult to debug when something goes wrong at 2am with a hardware project.
The core operation — "read files, build prompt, call LLM, display response" — is 50 lines of Python.
Do not add a framework to do what 50 lines can do.

**Revisit:** Phase 3+ if multi-agent orchestration is genuinely needed.
