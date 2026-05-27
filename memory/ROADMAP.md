# Zeno Workshop — Roadmap

_This is not a timeline. It is a dependency graph. Each phase produces a working demo before the next starts._

---

## Phase 1 — Project-Aware Assistant (CURRENT)

**Goal:** Read a local project folder and answer engineering questions about it.

**Demo:** CLI tool that loads any Arduino/ESP32/Python project and can:
- Show file tree
- Summarize the project
- Answer questions about code
- Identify errors in logs
- Save session notes

**Exit criteria:**
- [ ] Tested with at least 3 different real project folders
- [ ] At least one real bug diagnosed correctly
- [ ] GitHub commit with demo video
- [ ] Someone else can clone and run it in under 10 minutes

**Known risks:**
- Ollama not installed on target machine → documented in README
- LLM hallucinating incorrect function names → warn users in README
- Context window too small for large projects → tune max_chars

---

## Phase 2 — Live Debug Assistant

**Goal:** Add real-time serial/log monitoring and voice input.

**New capabilities:**
- Read Arduino serial output in real time (pyserial)
- Watch log files for changes (watchdog)
- Accept voice commands (Whisper STT, local)
- Speak responses (pyttsx3 or piper TTS)
- Auto-trigger analysis when errors appear in serial/log

**Exit criteria:**
- [ ] Demonstrated live during actual ESP32 debugging session
- [ ] Voice command recognized correctly in 8/10 attempts
- [ ] Serial error triggers automatic Zeno analysis

---

## Phase 3 — HUD Card Interface

**Goal:** Replace CLI with a persistent desktop overlay showing project status, recent queries, and live serial output.

**New capabilities:**
- Floating HUD window (transparent, stays on top)
- Card-based layout: project status, last response, serial tail, notes
- Camera feed thumbnail (USB camera, OpenCV)
- Click-to-query mode

**Exit criteria:**
- [ ] HUD visible while working in Arduino IDE or VSCode
- [ ] Camera thumbnail updating in real time
- [ ] Last 5 serial lines visible on HUD

---

## Phase 4 — ESP32 Glove Control

**Goal:** Use ESP32 + MPU6050 gesture glove to control Zeno commands.

**New capabilities:**
- BLE or ESP-NOW bridge from glove to Zeno Python
- Gesture → command mapping (configurable)
- Hands-free trigger for voice query
- Gesture scroll through HUD cards

**Exit criteria:**
- [ ] At least 5 gestures reliably recognized
- [ ] One gesture triggers voice query
- [ ] Latency < 200ms glove → HUD response

---

## Phase 5 — AR Glasses HUD Bridge

**Goal:** Push Zeno's HUD cards to monocular AR glasses display.

**New capabilities:**
- Protocol design for AR display (serial/BLE/WiFi)
- Minimal data format for glasses (bandwidth constrained)
- Context-aware display: show only what matters right now
- Hands-free mode: glove + glasses + voice only

**Exit criteria:**
- [ ] At least 2 HUD card types rendered on glasses
- [ ] Full loop: voice query → Zeno response → glasses display
- [ ] Functional during actual hardware debug session

---

## Phase 6 — Public Launch Package

**Goal:** Package Zeno Workshop as a public-facing project with full documentation and demo.

**Deliverables:**
- GitHub repository with clean README, setup guide, and demo video
- Build log series (written during phases 1-5)
- Demo video showing each phase working
- Hardware BOM with prices and sources
- Community: Discord or GitHub Discussions

**This is the reputation and portfolio artifact.**

---

## Phases Beyond Scope (Do Not Plan Yet)

- Multi-room deployment
- Cloud sync
- Commercial product
- SaaS
- Anything requiring significant upfront money

_If the project gets traction, these decisions will make themselves._
