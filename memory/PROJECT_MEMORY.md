# Zeno Workshop — Project Memory

_Last updated: 2026-05-27_

---

## What Is Zeno Workshop

A room-built AI assistant for makers and embedded engineers.
Running locally on personal hardware, accessing a physical workspace through camera, microphone, file system, and serial port.
Not a cloud product. Not a SaaS. A tool built from scraps that documents its own construction publicly.

Long-term vision:
- See the electronics desk (camera)
- Hear voice commands (Whisper STT)
- Read active project files, logs, serial output in real time
- Debug Arduino/ESP32/Python/Linux projects
- Remember decisions and failures across sessions
- Display information as HUD cards
- Accept input from ESP32 gesture glove (MPU6050, BLE/ESP-NOW)
- Bridge into monocular AR glasses HUD

---

## Current Goal

**Demo 001 — Project-Aware Zeno**
A CLI assistant that reads a local project folder and answers engineering questions about it using a local Ollama LLM.

---

## Current Hardware

- Primary development machine: Windows PC (ZenoFinal workspace)
- ESP32 (DevKit V1) — gesture glove prototype
- MPU6050 — IMU on glove
- Raspberry Pi (available) — potential server/bridge
- AR monocular glasses — future HUD target
- USB camera — available, not yet integrated

---

## Current Software Stack

| Component      | Tool/Library              | Status          |
|---------------|---------------------------|-----------------|
| Language       | Python 3.10+              | Active          |
| LLM backend    | Ollama (local)            | Requires setup  |
| LLM model      | mistral (configurable)    | Requires pull   |
| Terminal UI    | rich                      | Active          |
| Config         | PyYAML                    | Active          |
| HTTP client    | requests                  | Active          |
| STT            | Whisper (local)           | Phase 2         |
| TTS            | pyttsx3 / piper           | Phase 2         |
| Camera input   | OpenCV                    | Phase 3         |
| Glove bridge   | BLE / ESP-NOW + Python    | Phase 4         |
| AR HUD         | TBD                       | Phase 5         |

---

## Important Decisions

See DECISIONS.md

---

## What Failed

See FAILURES.md

---

## What Worked

- Demo 001 structure designed and scaffolded (2026-05-27)
- Full project scanner working (scan_project, build_context_text)
- Ollama integration via HTTP /api/generate
- Rich terminal card output
- Memory file system (markdown append)

---

## Next Milestone

**Demo 001 — verified working**
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Pull Ollama model: `ollama pull mistral`
- [ ] Run: `python run.py ./test_project`
- [ ] Ask 3 engineering questions about test_project
- [ ] Save a session note
- [ ] Run :summarize and verify output
- [ ] Commit to GitHub with working demo video
