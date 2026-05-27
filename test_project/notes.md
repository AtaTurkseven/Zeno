# Project Notes — ESP32 MPU6050 Logger

## 2026-05-26

### Problem: DMP init always fails (error code 1)

Tried:
- Different I2C speeds (100kHz, 400kHz)
- Separate 3.3V regulator for MPU6050
- 4.7kΩ pull-up resistors on SDA/SCL
- Different MPU6050 library versions

Still failing. DMP error code 1 = failed to write DMP firmware to MPU6050 internal memory.

Possible causes I haven't ruled out:
- I2C clock stretching interaction with ESP32 hardware I2C
- BLE WiFi causing I2C interference? (radio noise on power rail?)
- Library bug with this specific MPU6050 clone chip (common cheap Chinese clone may have different firmware checksum)

Workaround for now: skip DMP, use raw accel/gyro + software tilt calculation.

### Problem: Crashes after BLE disconnect (~28s)

The watchdog reset happens roughly 28 seconds after BLE client disconnects.
Backtrace points to ISR stack.

Suspected cause: vTaskDelay() call inside the ISR (I put it there to debounce, should use a flag instead).

Next attempt: remove vTaskDelay from ISR, use volatile bool flag + debounce in loop().

### Wiring confirmed working:
```
MPU6050 VCC → ESP32 3.3V (NOT 5V pin)
MPU6050 GND → ESP32 GND
MPU6050 SDA → ESP32 GPIO 21
MPU6050 SCL → ESP32 GPIO 22
MPU6050 INT → ESP32 GPIO 2
AD0 → GND (I2C address 0x68)
```
