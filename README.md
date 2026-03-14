# Zeno
Zeno is a modular physical AI assistant system with robotic interaction, local + cloud intelligence, and hardware control capabilities designed for real-world engineering projects.

---

## Features

| Area | Status | Notes |
|------|--------|-------|
| Local AI backend | ✅ | Ollama-compatible servers |
| Cloud AI backend | ✅ | OpenAI & Anthropic |
| Vector memory | ✅ | In-memory cosine-similarity store |
| Tool system | ✅ | Registry + executor; built-in calculator, time, echo |
| CLI interface | ✅ | Full REPL with `/tool`, `/memory`, `/status` |
| GUI interface | ✅ | Tkinter chat window (stdlib — no extra deps) |
| ESP32 driver | ✅ | Text-based UART protocol |
| Robotic arm driver | ✅ | 6-DOF serial servo controller |
| Sensor manager | ✅ | Multi-sensor aggregation with caching |
| Camera (vision) | ✅ | OpenCV capture with graceful fallback |
| Voice listener | ✅ | SpeechRecognition / Google STT with fallback |
| Voice speaker | ✅ | pyttsx3 offline TTS with fallback |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Optional hardware/voice/vision extras:

```bash
# Camera capture
pip install opencv-python-headless>=4.8

# Voice (speech recognition + text-to-speech)
pip install SpeechRecognition pyaudio pyttsx3
```

### 2. Configure

Copy and edit the settings file:

```bash
cp config/settings.yaml config/local_settings.yaml
# Edit ai.backend, devices.*, voice.enabled, etc.
```

### 3. Run

```bash
# CLI mode (default)
python main.py

# GUI mode (Tkinter)
python main.py --ui gui

# Use a custom config
python main.py --config config/local_settings.yaml --ui cli
```

---

## Architecture

```
zeno/
├── core/
│   ├── ai/          # AIBase, LocalLLM (Ollama), CloudAI (OpenAI/Anthropic)
│   ├── memory/      # VectorStore — in-memory cosine-similarity search
│   ├── tools/       # BaseTool, ToolRegistry, ToolExecutor
│   │   └── builtin/ # CalculatorTool, TimeTool, EchoTool
│   ├── hardware/    # BaseDevice, SerialDevice
│   ├── vision/      # Camera (OpenCV)
│   └── voice/       # VoiceListener (STT), VoiceSpeaker (TTS)
├── devices/
│   ├── esp32/       # ESP32Controller
│   ├── robotic_arm/ # RoboticArm (6-DOF)
│   └── sensors/     # SensorManager
├── ui/
│   ├── cli/         # CLIShell — interactive REPL
│   └── gui/         # GUIApp — Tkinter chat window
└── config/          # ConfigManager (YAML, dot-notation)
```

### Adding a Custom Tool

```python
from zeno.core.tools.base import BaseTool, ToolResult

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(name="my_tool", description="Does something useful.")

    @property
    def parameters(self):
        return {"value": {"type": "string", "description": "Input value."}}

    def execute(self, **kwargs):
        return ToolResult(success=True, output=kwargs.get("value", ""))
```

Register it in `main.py` inside `_build_tool_system()`:

```python
registry.register(MyTool())
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Show AI availability and memory size |
| `/tool <name> [key=value …]` | Run a registered tool |
| `/memory <query>` | Search vector memory |
| `/quit` | Exit |

Any other input is forwarded to the AI backend.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Hardware Setup

| Device | Default Port | Baud Rate |
|--------|-------------|-----------|
| ESP32 | `/dev/ttyUSB0` | 115200 |
| Robotic Arm | `/dev/ttyUSB1` | 9600 |

Detect connected serial devices:

```bash
python scripts/check_devices.py
```
