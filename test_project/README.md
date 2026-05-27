# ESP32 MPU6050 Logger

An ESP32-based project that reads accelerometer and gyroscope data from an MPU6050 IMU sensor over I2C, logs it to serial, and pushes alerts over BLE when tilt exceeds a threshold.

## Status

**IN PROGRESS — NOT STABLE**

- Basic I2C communication works intermittently
- BLE notifications crash firmware after ~30 seconds
- DMP mode never initialized successfully

## Hardware

- ESP32 DevKit V1
- MPU6050 (I2C address: 0x68)
- 3.3V power (not 5V — learned this the hard way)
- SDA → GPIO 21
- SCL → GPIO 22

## Dependencies

- Wire.h (built-in)
- MPU6050 by Electronic Cats
- BLEDevice (ESP32 Arduino Core built-in)

## Known Issues

1. `MPU6050_dmp_init()` returns error code 1 (DMP initialization failed)
2. BLE disconnect causes hard fault → reboot
3. I2C address conflict suspected when BLE stack active
4. `vTaskDelay` call inside interrupt handler (DO NOT DO THIS — left as a bug to find)

## Serial Output

115200 baud

## Build

Arduino IDE 2.x, ESP32 board package 2.0.x
