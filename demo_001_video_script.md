# Demo 001 — Video Script
## "Zeno Workshop: Project-Aware AI Assistant"

**Format:** screen recording, terminal visible, no face required
**Length:** 90–120 seconds
**Target platform:** YouTube, GitHub, Twitter/X

---

## Opening (0:00–0:10)

**[No intro card needed. Start directly in terminal.]*

Show the terminal. Type:

```bash
python run.py ./test_project
```

Let Zeno boot. The banner and project summary card appear.

**Voiceover (or caption):**
> "This is Zeno Workshop. It reads your project folder and lets you ask engineering questions about it. Runs locally. No internet."

---

## Scene 1 — Project Load (0:10–0:25)

Project summary card is on screen. Show:
- Project name: `test_project`
- Type: `Arduino/ESP32`
- Files loaded: 4
- Path

Type: `:tree`

File tree appears.

**Voiceover:**
> "It loaded an ESP32 project with an IMU sensor. Four files — the sketch, logs, notes, and README."

---

## Scene 2 — Real Question (0:25–0:55)

Type (slowly, legibly):

```
Why is the firmware crashing after BLE disconnect?
```

Wait for response. Zeno should identify:
- The `vTaskDelay()` call inside the ISR on line ~110
- The missing `startAdvertising()` call on disconnect
- Connect it to the watchdog reset in the error log

**Voiceover:**
> "It found the bug. vTaskDelay inside an interrupt handler is not safe in FreeRTOS. And it found a second issue — BLE won't reconnect after disconnect because advertising isn't restarted."

---

## Scene 3 — Log Analysis (0:55–1:15)

Type:

```
What does the Guru Meditation error in errors.log mean and what caused it?
```

Zeno should reference:
- `Interrupt wdt timeout on CPU0`
- ISR blocking too long
- vTaskDelay as likely culprit

**Voiceover:**
> "Watchdog timeout on CPU0. The ISR was blocked by vTaskDelay and the watchdog reset the chip. Exact root cause, file and line."

---

## Scene 4 — Save a Note (1:15–1:25)

Type:

```
:note Remove vTaskDelay from ISR. Use volatile flag. Add startAdvertising() on disconnect callback.
```

Show status: "Note saved → memory/SESSION_NOTES.md"

**Voiceover:**
> "Notes save to markdown files. Persistent across sessions. No database."

---

## Closing (1:25–1:35)

Type `:summarize`. Show the summary card briefly.

**Voiceover or caption:**
> "Zeno Workshop. Phase 1 of 6. Next: serial port monitoring and voice input."

**End card (text overlay):**
```
github.com/[your-handle]/zeno-workshop
Local AI. No cloud. Built from scraps.
```

---

## Recording notes

- Use a terminal with dark background (Windows Terminal with dark theme)
- Font size: at least 16pt so it's readable in video
- Zoom in if needed — the rich cards should fill the frame
- Record at 1920x1080 minimum
- You do not need to speak. Captions are enough for a first demo.
- Keep it under 2 minutes. Respect the viewer's time.
- Do NOT add music, logo animations, or a 30-second intro. It is cringe and it undermines credibility.

---

## Upload checklist

- [ ] Video is under 2 minutes
- [ ] Title: "Zeno Workshop Demo 001 — Project-Aware AI Assistant (local, no cloud)"
- [ ] Description: paste README intro paragraph + link to GitHub
- [ ] GitHub repo is public before posting
- [ ] BUILD_LOG.md entry written
