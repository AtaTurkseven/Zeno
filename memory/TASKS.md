# Zeno Workshop — Tasks

_Updated: 2026-05-27_

---

## TODAY

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify Ollama is installed: `ollama --version`
- [ ] Pull mistral model: `ollama pull mistral` (or smaller: `ollama pull phi3`)
- [ ] Run Demo 001: `python run.py ./test_project`
- [ ] Verify project loads and tree displays correctly
- [ ] Ask 3 real questions about test_project, verify LLM response quality
- [ ] Test `:note` command, verify SESSION_NOTES.md is written
- [ ] Test `:summarize` command, verify PROJECT_SUMMARIES.md is written
- [ ] Fix any bugs found during first run
- [ ] Commit initial structure to GitHub: `git init && git add . && git commit -m "feat: Demo 001 bootstrap"`

---

## THIS WEEK (Days 2-7)

**Day 2 — Real project test**
- [ ] Load a real Arduino or ESP32 project you have on disk
- [ ] Ask Zeno about a real bug or a real design question
- [ ] Compare Zeno's answer to what you actually know — is it accurate?
- [ ] Document any hallucinations in FAILURES.md

**Day 3 — Error/log detection**
- [ ] Add a test with a real errors.log containing stack traces
- [ ] Verify Zeno can identify the error source file and line
- [ ] Ask: "What is causing this error and how do I fix it?"

**Day 4 — Model comparison**
- [ ] Try at least 2 models: mistral, phi3, codellama (if disk space allows)
- [ ] Document which model gives better embedded/Arduino answers
- [ ] Update TECH_STACK.md with findings

**Day 5 — Context quality**
- [ ] Load a larger project (>10 files)
- [ ] Test if the 14000-char context limit causes missed info
- [ ] Tune max_chars in scanner.py if needed

**Day 6 — Polish + README**
- [ ] Clean up any rough output formatting in cards.py
- [ ] Finalize README.md with actual tested setup instructions
- [ ] Record terminal session for demo video

**Day 7 — Public commit**
- [ ] Push to GitHub (public repo)
- [ ] Write first public build log post (can use BUILD_LOG.md entry as base)
- [ ] Post short demo video (screen recording is fine)

---

## LATER

- [ ] Voice input (Whisper STT) — Phase 2
- [ ] TTS output (pyttsx3 or piper) — Phase 2
- [ ] Serial port reader (pyserial) — real-time Arduino debug — Phase 2
- [ ] Camera input (OpenCV) — Phase 3
- [ ] HUD card UI (tkinter or pygame overlay) — Phase 3
- [ ] ESP32 glove BLE/ESP-NOW bridge — Phase 4
- [ ] AR glasses HUD protocol design — Phase 5
- [ ] Vector memory (Chroma/Qdrant) for long-term project memory — Phase 3+
- [ ] Multi-project awareness (index multiple folders) — Phase 2+

---

## BLOCKED

_Nothing blocked yet — record blockers here with reason_

- Example format: `- [ ] [BLOCKED: need hardware X] Task description`
