# Zeno Workshop — Tech Stack

_Only list tools that have been tested and verified. Mark untested as [PLANNED]._

---

## Python Packages

| Package     | Version  | Purpose                        | Status    |
|-------------|----------|--------------------------------|-----------|
| rich        | >=13.0   | Terminal UI (cards, panels)    | Active    |
| requests    | >=2.28   | HTTP client for Ollama API     | Active    |
| pyyaml      | >=6.0    | Config file parsing            | Active    |
| re          | stdlib   | Local issue/log pattern scan   | Active    |
| pyserial    | >=3.5    | Arduino serial reading         | PLANNED   |
| whisper     | latest   | Local STT (OpenAI Whisper)     | PLANNED   |
| pyttsx3     | >=2.90   | Local TTS (offline)            | PLANNED   |
| opencv-python | >=4.8  | Camera input                  | PLANNED   |
| watchdog    | >=3.0    | File change monitoring         | PLANNED   |

---

## Ollama Models

| Model                  | Size   | Use case                           | Status    |
|------------------------|--------|------------------------------------|-----------|
| malixator/ZenoV1       | 7.2 GB | **Configured now** — custom Zeno model | Active |
| qwen3:8b               | 5.2 GB | Strong at code/embedded            | Available |
| mistral:7b             | 4.4 GB | General + code                     | Available |
| gemma3:4b              | 3.3 GB | Fast, lighter CPU load             | Available |
| qwen3:1.7b             | 1.4 GB | Fast responses, weak at deep code  | Available |
| zeno-gemma4-e4b:latest | 9.6 GB | Custom Zeno model — test it        | Available |
| gemma3:270m            | 291 MB | Tiny test model                    | Available |

---

## STT / TTS

| Tool       | Type  | Model/Backend     | Offline? | Status   |
|------------|-------|-------------------|----------|----------|
| Whisper    | STT   | tiny / base / small | Yes    | PLANNED  |
| pyttsx3    | TTS   | OS voice engine   | Yes      | PLANNED  |
| piper      | TTS   | ONNX neural TTS   | Yes      | PLANNED  |

---

## UI Framework

| Layer      | Tool              | Status    |
|------------|-------------------|-----------|
| Phase 1    | rich (terminal HUD cards) | Active |
| Phase 3    | TBD (tkinter / pygame / electron) | PLANNED |
| Phase 5    | Custom AR protocol | PLANNED  |

---

## Hardware Boards

| Board           | Role                        | Protocol         | Status    |
|-----------------|-----------------------------|------------------|-----------|
| ESP32 DevKit V1 | Gesture glove MCU           | BLE / ESP-NOW    | Available |
| MPU6050         | IMU on glove                | I2C              | Available |
| Raspberry Pi    | Potential inference server  | SSH / WiFi       | Available |
| USB camera      | Desk visual input           | V4L2 / OpenCV    | Available |
| AR monocular glasses | HUD display target    | TBD              | Available |

---

## Communication Protocols

| Protocol  | Use case                                | Status    |
|-----------|-----------------------------------------|-----------|
| HTTP/REST | Ollama API calls                        | Active    |
| BLE       | ESP32 glove → PC bridge                 | PLANNED   |
| ESP-NOW   | ESP32 glove → ESP32 receiver (faster)  | PLANNED   |
| Serial    | Arduino debug output → Zeno             | PLANNED   |
| WebSocket | Live log streaming (Phase 2+)           | PLANNED   |

---

## Known Issues

_Record confirmed bugs and limitations here_

- Ollama response time on CPU: 30-90s for mistral-7B. Acceptable for Phase 1. Will need optimization or model switch for voice mode.
- Context window: hard-capped at 14000 chars in scanner.py. Large projects will be truncated. Monitor and tune.
- Windows path handling: scanner.py uses pathlib — should be cross-platform but test on Windows explicitly.
- Deterministic analyzer is heuristic-based. It is strong for obvious issues and logs, but not a substitute for real semantic code understanding.
