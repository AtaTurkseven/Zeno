# Zeno Workshop — Failure Log

_Record every failure here. No shame. This is the most valuable document in the project._

---

## Template

```
### [DATE] — [WHAT FAILED]

**Symptoms:**
What was observed. Error messages, unexpected behavior, hardware behavior.

**Root cause:**
If known. If not known, write "Unknown — suspected: [guess]"

**Fix attempted:**
What was tried and whether it worked.

**Final lesson:**
One sentence takeaway.
```

---

## 2026-05-27 — Blank Answer For Valid HUD Question

**Symptoms:**
When testing Zeno against the real Zeno workspace, the question "How do I launch the HUD prototype?" returned a blank answer.

**Root cause:**
`query_ollama()` treated an empty model response as valid text instead of a failure, so the normal fallback logic never ran.

**Fix attempted:**
Changed `zeno/llm.py` so empty responses return `[EMPTY RESPONSE] Model returned no text.` and added deterministic fallback answers in `zeno/analyzer.py` for HUD launch and memory-location questions.

**Final lesson:**
An empty model response is still a failure and must trigger fallback behavior.

---

## 2026-05-27 — No failures recorded yet

Project bootstrapped. Demo 001 not yet tested on real hardware.
First failures will appear after the first real run.

---

_Add entries above this line as failures occur._
