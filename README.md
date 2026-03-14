# Zeno — Modular Physical AI Assistant

Zeno is an open-source, modular AI assistant designed to bridge software intelligence with physical hardware. It combines a conversational AI backend with device control (ESP32, robotic arm, sensors), voice I/O, and computer vision in a single extensible Python framework.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Zeno](#running-zeno)
- [AI Backends](#ai-backends)
- [Device Support](#device-support)
- [Voice & Vision](#voice--vision)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Features

- **Dual AI backends** — run a local LLM via [Ollama](https://ollama.com/) or connect to OpenAI / Anthropic cloud APIs
- **Episodic memory** — in-memory vector store with cosine-similarity search, with a path to upgrade to [Chroma](https://www.trychroma.com/)
- **Extensible tool system** — register callable tools at runtime; the executor wraps them with error handling and timing
- **Hardware device drivers** — serial-based drivers for ESP32, a multi-DOF robotic arm, and an arbitrary sensor manager
- **Interactive CLI** — a REPL shell with slash-commands (`/help`, `/status`, `/tool`, `/memory`, `/quit`)
- **GUI placeholder** — a stub `GUIApp` that falls back to the CLI shell until a full GUI is implemented
- **Voice I/O (planned)** — text-to-speech and speech-to-text interfaces are defined; synthesis and capture are marked as TODOs
- **Computer vision (planned)** — `Camera` dataclass and frame-capture interface are ready; OpenCV integration is a TODO

---

## Architecture

```
main.py
  ├── ConfigManager          (config/settings.yaml)
  ├── Logging setup
  ├── AI Backend             (LocalLLM via Ollama  OR  CloudAI via OpenAI/Anthropic)
  ├── VectorStore            (in-memory episodic memory, cosine-similarity search)
  ├── ToolRegistry + ToolExecutor
  └── UI
        ├── CLIShell         (implemented — interactive REPL)
        └── GUIApp           (stub — falls back to CLI)
```

Key package layout inside `zeno/`:

| Package | Responsibility |
|---|---|
| `zeno.config` | YAML configuration loading with dot-notation access |
| `zeno.core.ai` | AI backend abstraction (`AIBase`), `LocalLLM`, `CloudAI` |
| `zeno.core.memory` | `VectorStore` with NumPy-backed cosine search |
| `zeno.core.tools` | `BaseTool`, `ToolRegistry`, `ToolExecutor` |
| `zeno.core.hardware` | `BaseDevice`, `SerialDevice` base classes |
| `zeno.core.voice` | `VoiceSpeaker` (TTS) and `VoiceListener` (STT) interfaces |
| `zeno.core.vision` | `Camera` and `Frame` dataclass |
| `zeno.devices.esp32` | `ESP32Controller` over serial UART |
| `zeno.devices.robotic_arm` | `RoboticArm` with per-joint angle clamping |
| `zeno.devices.sensors` | `SensorManager` for polling named sensor callables |
| `zeno.ui.cli` | `CLIShell` REPL |
| `zeno.ui.gui` | `GUIApp` stub |

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | 3.12 recommended; uses `str \| None` union syntax |
| pip | any recent | |
| Ollama | any | Required only for the `local` AI backend |

**Optional** (for planned features):
- `opencv-python` — computer vision / camera capture
- `SpeechRecognition` — speech-to-text
- `pyttsx3` or `gTTS` — text-to-speech
- A GUI toolkit (Tkinter, PyQt6, Gradio, or Streamlit) — once the GUI is implemented

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AtaTurkseven/Zeno.git
cd Zeno

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Local LLM only) Start Ollama and pull a model
ollama serve &
ollama pull llama3
```

---

## Configuration

All settings live in `config/settings.yaml`. Key sections:

```yaml
system:
  name: Zeno
  version: "0.1.0"
  log_level: INFO          # DEBUG | INFO | WARNING | ERROR
  log_file: logs/zeno.log  # omit to log to console only

ai:
  backend: local           # "local" (Ollama) or "cloud" (OpenAI/Anthropic)
  local:
    model: llama3
    host: http://localhost:11434
  cloud:
    provider: openai       # "openai" or "anthropic"
    api_key_env: OPENAI_API_KEY
    model: gpt-4o

memory:
  embedding_dim: 384
  max_entries: 10000

devices:
  esp32:
    port: /dev/ttyUSB0
    baud_rate: 115200
  robotic_arm:
    port: /dev/ttyUSB1
    baud_rate: 9600
    dof: 6

voice:
  enabled: false           # TTS/STT not yet implemented
vision:
  enabled: false           # Camera capture not yet implemented

ui:
  mode: cli                # "cli" or "gui"
```

For the cloud backend, export your API key before running:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Running Zeno

```bash
# Default: reads config/settings.yaml, launches configured UI mode
python main.py

# Override the config file path
python main.py --config /path/to/my_settings.yaml

# Override the UI mode at launch time
python main.py --ui cli
python main.py --ui gui      # falls back to CLI until GUI is implemented
```

### CLI Shell Commands

Once the CLI shell is running, the following slash-commands are available:

| Command | Description |
|---|---|
| `/help` | List available commands |
| `/status` | Show status of all subsystems |
| `/tool <name>` | Execute a registered tool by name |
| `/memory <query>` | Search the vector memory store |
| `/quit` | Exit Zeno (also `Ctrl-D`) |

Any other input is forwarded directly to the AI backend as a prompt.

### Device Connectivity Check

```bash
python scripts/check_devices.py
```

---

## AI Backends

### Local — Ollama

Zeno sends HTTP requests to a locally running Ollama server.

```yaml
ai:
  backend: local
  local:
    model: llama3                    # any model pulled with `ollama pull`
    host: http://localhost:11434
    timeout: 60
```

Start Ollama before launching Zeno:

```bash
ollama serve
```

### Cloud — OpenAI / Anthropic

```yaml
ai:
  backend: cloud
  cloud:
    provider: openai          # or "anthropic"
    api_key_env: OPENAI_API_KEY
    model: gpt-4o
    timeout: 30
```

---

## Device Support

### ESP32

`ESP32Controller` communicates over a serial UART using ASCII command strings.

```python
from zeno.devices.esp32.controller import ESP32Controller

esp = ESP32Controller(port="/dev/ttyUSB0", baud_rate=115200)
esp.connect()
esp.set_led(True)           # turn on built-in LED
value = esp.read_pin(34)    # read ADC value (0–4095)
esp.reset()
esp.disconnect()
```

### Robotic Arm

`RoboticArm` controls a multi-DOF serial arm with per-joint angle clamping (0–180°).

```python
from zeno.devices.robotic_arm.arm import RoboticArm

arm = RoboticArm(port="/dev/ttyUSB1", baud_rate=9600, dof=6)
arm.connect()
arm.home()                   # move all joints to 90°
arm.move_joint(0, 45)        # move joint 0 to 45°
position = arm.get_position()
arm.disconnect()
```

### Sensor Manager

`SensorManager` polls arbitrary named sensor callables and caches their latest readings.

```python
from zeno.devices.sensors.sensor_manager import SensorManager

manager = SensorManager()
manager.register("temperature", lambda: read_temp_sensor())
manager.register("humidity",    lambda: read_humidity_sensor())

readings = manager.read_all()   # {"temperature": 23.5, "humidity": 61.2}
manager.unregister("humidity")
```

---

## Voice & Vision

> **Status:** The interfaces are fully defined, but the underlying implementations (audio synthesis/capture and camera integration) are marked as TODOs.

### Voice

| Component | Class | Status |
|---|---|---|
| Text-to-speech | `VoiceSpeaker` | Interface ready; `_synthesise()` / `_play()` are TODOs |
| Speech-to-text | `VoiceListener` | Interface ready; `_capture_audio()` / `_transcribe_audio()` are TODOs |

Enable in config (`voice.enabled: true`) and install an audio library (e.g. `pyttsx3`, `SpeechRecognition`) when implementations land.

### Vision

| Component | Class | Status |
|---|---|---|
| Camera capture | `Camera` / `Frame` | Interface ready; `_open_capture()` / `_read_frame()` are TODOs (OpenCV) |

Enable in config (`vision.enabled: true`) and install `opencv-python` when the implementation lands.

---

## Testing

```bash
# Install test runner
pip install pytest

# Run the full test suite
python -m pytest tests/ -v
```

Tests are organised by subsystem:

| File | Coverage |
|---|---|
| `tests/test_config.py` | ConfigManager — YAML loading, nested key access, defaults |
| `tests/test_core_ai.py` | AIBase interface, LocalLLM / CloudAI mocking |
| `tests/test_core_memory.py` | VectorStore — add, search, eviction |
| `tests/test_core_tools.py` | ToolRegistry / ToolExecutor — registration, execution, error handling |
| `tests/test_devices.py` | SensorManager, device protocol |

---

## Project Structure

```
Zeno/
├── main.py                        # Entry point
├── requirements.txt               # Core Python dependencies
├── config/
│   └── settings.yaml              # Main configuration file
├── scripts/
│   └── check_devices.py           # Serial port connectivity checker
├── tests/
│   ├── test_config.py
│   ├── test_core_ai.py
│   ├── test_core_memory.py
│   ├── test_core_tools.py
│   └── test_devices.py
└── zeno/
    ├── config/
    │   └── manager.py             # ConfigManager
    ├── core/
    │   ├── ai/
    │   │   ├── base.py            # AIBase abstract class
    │   │   ├── local_llm.py       # Ollama integration
    │   │   └── cloud_ai.py        # OpenAI / Anthropic integration
    │   ├── memory/
    │   │   └── vector_store.py    # In-memory vector store
    │   ├── tools/
    │   │   ├── base.py            # BaseTool / ToolResult
    │   │   ├── registry.py        # ToolRegistry
    │   │   └── executor.py        # ToolExecutor
    │   ├── hardware/
    │   │   ├── base.py            # BaseDevice / DeviceStatus
    │   │   └── serial_device.py   # SerialDevice base class
    │   ├── voice/
    │   │   ├── speaker.py         # VoiceSpeaker (TTS interface)
    │   │   └── listener.py        # VoiceListener (STT interface)
    │   ├── vision/
    │   │   └── camera.py          # Camera / Frame
    │   └── logging_setup.py       # Logging configuration helper
    ├── devices/
    │   ├── esp32/
    │   │   └── controller.py      # ESP32Controller
    │   ├── robotic_arm/
    │   │   └── arm.py             # RoboticArm
    │   └── sensors/
    │       └── sensor_manager.py  # SensorManager
    └── ui/
        ├── cli/
        │   └── shell.py           # CLIShell (interactive REPL)
        └── gui/
            └── app.py             # GUIApp (stub)
```
