# Zeno Session Notes

### 2026-05-27 06:29
Project: test_project
Type: Arduino/ESP32
Loaded files: 4
Code files: 1 | Logs: 1 | Markdown: 2

Key files:
- ESP32_MPU6050_Logger.ino
- README.md
- errors.log
- notes.md

Detected issues:
- ISR contains vTaskDelay: notes.md documents vTaskDelay inside an ISR, which matches an interrupt watchdog reset pattern.
- Interrupt watchdog reset: errors.log shows an interrupt watchdog timeout, usually caused by ISR misuse or blocking work.
- MPU6050 DMP init failure: ESP32_MPU6050_Logger.ino indicates DMP initialization is failing; project is likely running in raw sensor fallback mode.

Next steps:
- Remove vTaskDelay or any blocking RTOS call from ISR context.
- Reproduce the reset while capturing serial logs after removing ISR blocking work.
- Keep raw accel/gyro fallback for now and isolate DMP init as a separate hardware/I2C debug task.

