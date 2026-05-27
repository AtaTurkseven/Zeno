# Zeno Interactions

## 2026-05-27 06:33 — test_project

### Query
why is the firmware crashing after ble disconnect

### Response
The firmware crashes due to a watchdog timeout, which is suspected to be caused by blocking operations within the interrupt context, as noted in `notes.md`.

**Diagnosis:**
The crash occurs when the BLE disconnect happens, suggesting an issue with the timing or execution flow related to the interrupt handling. The notes specifically point to the suspected cause: **a blocking call (like `vTaskDelay`) inside the Interrupt Service Routine (ISR)**.

**Actionable Steps:**

1.  **Review ISR:** Examine the `onMotionInterrupt()` function in `ESP32_MPU6050_Logger.ino`.
    ```cpp
    void IRAM_ATTR onMotionInterrupt() {
      motionDetected = true;
      // ISR must stay minimal. Never call blocking/RTOS delay APIs here.
    }
    ```
    The current implementation is minimal and does not contain blocking calls. However, the notes indicate this pattern is the source of the instability.

2.  **Implement Debouncing in Loop:** The code correctly handles debouncing in the main `loop()` context, which is the correct approach:
    ```cpp
    if (motionDetected) {
      // Debounce in normal task context to avoid ISR abuse and WDT resets.
      if (now - lastMotionHandledMs >= MOTION_DEBOUNCE_MS) {
        Serial.println("[MOTION] Motion interrupt fired.");
        lastMotionHandledMs = now;
      }
      motionDetected = false;

---

