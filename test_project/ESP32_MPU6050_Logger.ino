/*
  ESP32_MPU6050_Logger.ino
  Reads MPU6050 accel/gyro, logs to serial, sends BLE notifications on tilt alert.

  Hardware:
    ESP32 DevKit V1
    MPU6050 → SDA:GPIO21, SCL:GPIO22, VCC:3.3V, GND:GND
    I2C address: 0x68

  Known issues (see README):
    - DMP init fails (error code 1)
    - BLE crash after ~30s disconnect
    - Bug left intentionally: vTaskDelay inside ISR (line ~110)
*/

#include <Wire.h>
#include <MPU6050.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── Config ────────────────────────────────────────────────────────────────────
#define TILT_THRESHOLD   25.0f   // degrees — trigger BLE alert above this
#define SAMPLE_INTERVAL  100     // ms between samples
#define MOTION_DEBOUNCE_MS 20    // handle motion IRQ bursts in loop, not ISR
#define BLE_SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define BLE_CHARACTERISTIC_UUID "87654321-4321-4321-4321-cba987654321"
#define MOTION_INTERRUPT_PIN    2  // MPU6050 INT pin

// ── Globals ───────────────────────────────────────────────────────────────────
MPU6050 mpu;
BLEServer*         bleServer         = nullptr;
BLECharacteristic* bleCharacteristic = nullptr;
bool               bleConnected      = false;
volatile bool      motionDetected    = false;
int16_t            ax, ay, az, gx, gy, gz;
float              roll = 0.0f, pitch = 0.0f;
unsigned long      lastSampleMs = 0;
unsigned long      lastMotionHandledMs = 0;

// ── BLE Callbacks ──────────────────────────────────────────────────────────────
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    bleConnected = true;
    Serial.println("[BLE] Client connected");
  }
  void onDisconnect(BLEServer* pServer) override {
    bleConnected = false;
    Serial.println("[BLE] Client disconnected");
    pServer->startAdvertising();
    Serial.println("[BLE] Advertising restarted");
  }
};

// ── Motion Interrupt ISR ───────────────────────────────────────────────────────
void IRAM_ATTR onMotionInterrupt() {
  motionDetected = true;
  // ISR must stay minimal. Never call blocking/RTOS delay APIs here.
}

// ── Setup ──────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("[BOOT] ESP32 MPU6050 Logger starting...");

  Wire.begin(21, 22);
  Wire.setClock(400000);  // 400kHz fast mode

  Serial.println("[I2C] Scanning for MPU6050...");
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("[ERROR] MPU6050 connection failed. Check SDA/SCL wiring.");
    // BUG: no retry or halt — falls through and continues with broken sensor
  } else {
    Serial.println("[OK] MPU6050 connected.");
  }

  // DMP initialization — currently broken
  Serial.println("[DMP] Attempting DMP initialization...");
  uint8_t dmpStatus = mpu.dmpInitialize();
  if (dmpStatus != 0) {
    Serial.print("[ERROR] DMP init failed, error code: ");
    Serial.println(dmpStatus);
    // Error code 1 = DMP memory write failed
    // Possible cause: 3.3V rail noise, bad I2C signal quality
    // DMP disabled — falling back to raw accel/gyro
  }

  // Motion interrupt
  pinMode(MOTION_INTERRUPT_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(MOTION_INTERRUPT_PIN),
                  onMotionInterrupt, RISING);
  mpu.setMotionDetectionThreshold(10);
  mpu.setMotionDetectionDuration(5);
  mpu.setIntMotionEnabled(true);

  // BLE init
  BLEDevice::init("Zeno_Glove");
  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new MyServerCallbacks());

  BLEService* service = bleServer->createService(BLE_SERVICE_UUID);
  bleCharacteristic = service->createCharacteristic(
    BLE_CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  bleCharacteristic->addDescriptor(new BLE2902());
  service->start();

  BLEAdvertising* advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(BLE_SERVICE_UUID);
  advertising->start();
  Serial.println("[BLE] Advertising started. UUID: " BLE_SERVICE_UUID);

  Serial.println("[BOOT] Setup complete.");
}

// ── Loop ───────────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL) return;
  lastSampleMs = now;

  // Read raw IMU data
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Convert to g and degrees/s
  float ax_g = ax / 16384.0f;
  float ay_g = ay / 16384.0f;
  float az_g = az / 16384.0f;

  // Calculate roll and pitch (no gyro fusion — prone to drift)
  roll  = atan2(ay_g, az_g) * 57.2958f;
  pitch = atan2(-ax_g, sqrt(ay_g * ay_g + az_g * az_g)) * 57.2958f;

  Serial.print("[IMU] roll=");
  Serial.print(roll, 2);
  Serial.print(" pitch=");
  Serial.print(pitch, 2);
  Serial.print(" ax="); Serial.print(ax_g, 3);
  Serial.print(" ay="); Serial.print(ay_g, 3);
  Serial.print(" az="); Serial.println(az_g, 3);

  // Tilt alert via BLE
  if (bleConnected && (abs(roll) > TILT_THRESHOLD || abs(pitch) > TILT_THRESHOLD)) {
    char buf[64];
    snprintf(buf, sizeof(buf), "TILT:%.1f,%.1f", roll, pitch);
    bleCharacteristic->setValue((uint8_t*)buf, strlen(buf));
    bleCharacteristic->notify();
    Serial.print("[BLE] Tilt alert sent: ");
    Serial.println(buf);
  }

  if (motionDetected) {
    // Debounce in normal task context to avoid ISR abuse and WDT resets.
    if (now - lastMotionHandledMs >= MOTION_DEBOUNCE_MS) {
      Serial.println("[MOTION] Motion interrupt fired.");
      lastMotionHandledMs = now;
    }
    motionDetected = false;
  }
}
